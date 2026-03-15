import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import pwnagotchi.ui.faces as faces


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return '{%s}' % key


class VoiceWidgetManager:
    SUPPORTED_WIDGETS = frozenset(('channel', 'aps', 'pwnd', 'uptime'))
    WIDGET_ALIASES = {
        'pwnd': 'shakes',
    }

    def __init__(self, toolbox_root: str):
        self.toolbox_root = toolbox_root
        self._view = None
        self._locale_cache: Dict[str, Dict[str, Any]] = {}

    def bind_view(self, view: Any):
        self._view = view

    def apply(
        self,
        tool_id: str,
        tool_spec: Optional[Dict[str, Any]],
        result: Optional[Dict[str, Any]],
        runtime_config: Optional[Dict[str, Any]] = None,
    ):
        if self._view is None or not isinstance(result, dict):
            return

        events = result.get('events', {})
        if not isinstance(events, dict):
            return

        changed = False
        widgets = events.get('widgets', {})
        if isinstance(widgets, dict):
            changed = self._apply_widgets(widgets) or changed

        voice_event = str(events.get('voice_event') or '').strip()
        if voice_event:
            changed = self._apply_voice_event(
                tool_id,
                tool_spec or {},
                voice_event,
                events.get('voice_context', {}),
                runtime_config or {},
            ) or changed

        if changed:
            self._refresh_view()

    def _apply_widgets(self, widgets: Dict[str, Any]) -> bool:
        changed = False
        for raw_name, value in widgets.items():
            name = str(raw_name or '').strip()
            if name not in self.SUPPORTED_WIDGETS:
                continue

            view_key = self.WIDGET_ALIASES.get(name, name)
            rendered = self._render_widget_value(view_key, value)
            try:
                self._view.set(view_key, rendered)
                changed = True
            except Exception:
                logging.debug('[kali.voice] failed widget update %s=%s', view_key, rendered, exc_info=True)
        return changed

    def _apply_voice_event(
        self,
        tool_id: str,
        tool_spec: Dict[str, Any],
        voice_event: str,
        context: Any,
        runtime_config: Dict[str, Any],
    ) -> bool:
        face_payload = self._resolve_face_payload(tool_spec, voice_event)
        payload = dict(context) if isinstance(context, dict) else {}
        phrases = self._resolve_phrases(tool_id, tool_spec, voice_event, runtime_config)
        explicit_status = str(payload.get('status_text') or '').strip()
        if phrases:
            phrase = random.choice(phrases)
            text = self._render_phrase(phrase, payload)
        elif explicit_status:
            text = explicit_status
        else:
            return False

        try:
            face = face_payload.get('face')
            if face is not None:
                self._view.set('face', face)
            if 'fps' in face_payload:
                self._view.set('face_sequence_fps', face_payload.get('fps'))
            if 'duration' in face_payload:
                self._view.set('face_sequence_duration', face_payload.get('duration'))
            if 'looping' in face_payload:
                self._view.set('face_sequence_looping', face_payload.get('looping'))
            if 'face_sequence' in face_payload:
                self._view.set('face_sequence', face_payload.get('face_sequence'))
            self._view.set('status', text)
            return True
        except Exception:
            logging.debug('[kali.voice] failed voice update tool=%s event=%s', tool_id, voice_event, exc_info=True)
            return False

    def _resolve_phrases(
        self,
        tool_id: str,
        tool_spec: Dict[str, Any],
        voice_event: str,
        runtime_config: Dict[str, Any],
    ) -> List[str]:
        tool_dir = str(tool_spec.get('dir') or '')
        lang = self._lang(runtime_config)
        locale_map = self._load_locale_map(tool_dir, lang)
        locale_key = 'voice.%s' % voice_event
        locale_value = locale_map.get(locale_key)
        phrases = self._normalize_phrase_list(locale_value)
        if phrases:
            return phrases

        tool_manifest = tool_spec.get('tool', {}) if isinstance(tool_spec.get('tool', {}), dict) else {}
        voice_cfg = tool_manifest.get('voice', {}) if isinstance(tool_manifest.get('voice', {}), dict) else {}
        phrases = self._normalize_phrase_list(voice_cfg.get(voice_event))
        if not phrases:
            logging.debug('[kali.voice] no phrases for tool=%s event=%s lang=%s', tool_id, voice_event, lang)
        return phrases

    def _resolve_face_payload(self, tool_spec: Dict[str, Any], voice_event: str) -> Dict[str, Any]:
        tool_manifest = tool_spec.get('tool', {}) if isinstance(tool_spec.get('tool', {}), dict) else {}
        face_cfg = tool_manifest.get('voice_faces', {}) if isinstance(tool_manifest.get('voice_faces', {}), dict) else {}
        options = face_cfg.get(voice_event)
        if options in (None, ''):
            return {}

        if isinstance(options, dict):
            sequence = self._normalize_face_list(options.get('sequence'))
            if sequence:
                payload: Dict[str, Any] = {
                    'face': sequence[0],
                    'face_sequence': sequence,
                }
                for key in ('fps', 'duration', 'looping'):
                    if key in options:
                        payload[key] = options[key]
                return payload

            random_faces = self._normalize_face_list(options.get('random'))
            if random_faces:
                return {'face': random.choice(random_faces)}

            return {}

        if isinstance(options, list):
            options = self._normalize_face_list(options)
            if not options:
                return {}
            return {'face': random.choice(options)}

        return {'face': self._resolve_face_name(options)}

    @staticmethod
    def _normalize_face_list(value: Any) -> List[Any]:
        if not isinstance(value, list):
            return []
        resolved = []
        for item in value:
            face = VoiceWidgetManager._resolve_face_name(item)
            if face not in (None, ''):
                resolved.append(face)
        return resolved

    @staticmethod
    def _resolve_face_name(value: Any):
        if isinstance(value, str) and hasattr(faces, value):
            return getattr(faces, value)
        return value

    def _load_locale_map(self, tool_dir: str, lang: str) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for candidate in self._locale_candidates(lang):
            locale_path = os.path.join(tool_dir, 'locale', '%s.json' % candidate)
            data = self._read_locale_file(locale_path)
            if isinstance(data, dict):
                merged.update(data)
        return merged

    def _read_locale_file(self, path: str) -> Dict[str, Any]:
        if not path or not os.path.isfile(path):
            return {}
        if path in self._locale_cache:
            return self._locale_cache[path]

        try:
            with open(path, 'rt', encoding='utf-8') as fp:
                data = json.load(fp) or {}
                if not isinstance(data, dict):
                    data = {}
                self._locale_cache[path] = data
                return data
        except Exception:
            logging.debug('[kali.voice] failed reading locale file: %s', path, exc_info=True)
            self._locale_cache[path] = {}
            return {}

    @staticmethod
    def _locale_candidates(lang: str) -> List[str]:
        normalized = str(lang or 'en').replace('_', '-')
        candidates: List[str] = []
        for value in (normalized, normalized.split('-', 1)[0], 'en'):
            value = str(value or '').strip()
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    @staticmethod
    def _lang(runtime_config: Dict[str, Any]) -> str:
        main_cfg = runtime_config.get('main', {}) if isinstance(runtime_config.get('main', {}), dict) else {}
        return str(main_cfg.get('lang', 'en') or 'en')

    @staticmethod
    def _normalize_phrase_list(value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item not in (None, '')]

    @staticmethod
    def _render_phrase(template: str, context: Dict[str, Any]) -> str:
        try:
            return str(template).format_map(_SafeFormatDict(context))
        except Exception:
            logging.debug('[kali.voice] failed to render phrase: %s', template, exc_info=True)
            return str(template)

    @staticmethod
    def _render_widget_value(view_key: str, value: Any) -> str:
        if view_key == 'shakes':
            try:
                return str(int(value))
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    def _refresh_view(self):
        try:
            if hasattr(self._view, 'update'):
                try:
                    self._view.update(force=True)
                except TypeError:
                    self._view.update()
            elif hasattr(self._view, 'refresh'):
                self._view.refresh()
        except Exception:
            logging.debug('[kali.voice] failed to refresh view', exc_info=True)
