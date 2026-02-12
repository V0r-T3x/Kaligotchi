import os
import logging
import threading
from time import sleep
from datetime import datetime,timedelta
from pwnagotchi import plugins
from pwnagotchi.utils import StatusFile
from flask import render_template_string
from flask import jsonify
import prctl

TEMPLATE = """
{% extends "base.html" %}
{% set active_page = "plugins" %}
{% block title %}
    Session stats
{% endblock %}

{% block styles %}
    {{ super() }}
    <link rel="stylesheet" href="/css/jquery.jqplot.min.css"/>
    <link rel="stylesheet" href="/css/jquery.jqplot.css"/>
    <style>
        div.chart {
            height: 400px;
            width: 100%;
        }
        div#session {
            width: 100%;
        }
    </style>
{% endblock %}

{% block scripts %}
    {{ super() }}
     <script type="text/javascript" src="/js/jquery.jqplot.min.js"></script>
     <script type="text/javascript" src="/js/jquery.jqplot.js"></script>
     <script type="text/javascript" src="/js/plugins/jqplot.mobile.js"></script>
     <script type="text/javascript" src="/js/plugins/jqplot.json2.js"></script>
     <script type="text/javascript" src="/js/plugins/jqplot.dateAxisRenderer.js"></script>
     <script type="text/javascript" src="/js/plugins/jqplot.highlighter.js"></script>
     <script type="text/javascript" src="/js/plugins/jqplot.cursor.js"></script>
     <script type="text/javascript" src="/js/plugins/jqplot.enhancedLegendRenderer.js"></script>
{% endblock %}

{% block script %}
    $(document).ready(function(){
        var ajaxDataRenderer = function(url, plot, options) {
        var ret = null;
        $.ajax({
            async: false,
            url: url,
            dataType:"json",
            success: function(data) {
                ret = data;
            }
        });
        return ret;
        };

    function loadFiles(url, elm) {
        var data = ajaxDataRenderer(url);
        var x = document.getElementById(elm);
        $.each(data['files'], function( index, value ) {
            var option = document.createElement("option");
            option.text = value;
            x.add(option);
        });
    }

    function loadData(url, elm, title, fill) {
        var data = ajaxDataRenderer(url);
        if (!data || !data.values || data.values.length == 0) return;
        var hasData = false;
        for (var i = 0; i < data.values.length; i++) {
            if (data.values[i] && data.values[i].length > 0) {
                hasData = true;
                break;
            }
        }
        if (!hasData) return;
        var plot_os = $.jqplot(elm, data.values,{
        title: title,
        stackSeries: fill,
        seriesDefaults: {
            showMarker: !fill,
            fill: fill,
            fillAndStroke: fill
        },
        legend: {
            show: true,
            renderer: $.jqplot.EnhancedLegendRenderer,
            placement: 'outsideGrid',
            labels: data.labels,
            location: 's',
            rendererOptions: {
                numberRows: '2',
            },
            rowSpacing: '0px'
        },
        axes:{
            xaxis:{
                renderer:$.jqplot.DateAxisRenderer,
                tickOptions:{formatString:'%H:%M:%S'}
            },
            yaxis:{
                tickOptions:{formatString:'%.2f'}
            }
        },
        highlighter: {
            show: true,
            sizeAdjust: 7.5
        },
        cursor:{
            show: true,
            tooltipLocation:'sw'
        }
        }).replot({
        axes:{
            xaxis:{
                renderer:$.jqplot.DateAxisRenderer,
                tickOptions:{formatString:'%H:%M:%S'}
            },
            yaxis:{
                tickOptions:{formatString:'%.2f'}
            }
        }
        });
    }

    function loadSessionFiles() {
        loadFiles('/plugins/session-stats/session', 'session');
        $("#session").change(function() {
            loadSessionData();
        });
    }

    function loadSessionData() {
        var x = document.getElementById("session");
        var session = x.options[x.selectedIndex].text;
        loadData('/plugins/session-stats/os' + '?session=' + session, 'chart_os', 'OS', false)
        loadData('/plugins/session-stats/temp' + '?session=' + session, 'chart_temp', 'Temp', false)
        loadData('/plugins/session-stats/wifi' + '?session=' + session, 'chart_wifi', 'Wifi', true)
        loadData('/plugins/session-stats/duration' + '?session=' + session, 'chart_duration', 'Sleeping', true)
        loadData('/plugins/session-stats/reward' + '?session=' + session, 'chart_reward', 'Reward', false)
        loadData('/plugins/session-stats/epoch' + '?session=' + session, 'chart_epoch', 'Epochs', false)
        loadData('/plugins/session-stats/reflex_error' + '?session=' + session, 'chart_reflex_error', 'Reflex Errors', false)
        loadData('/plugins/session-stats/reflex_sys' + '?session=' + session, 'chart_reflex_sys', 'Reflex System', false)
        loadData('/plugins/session-stats/reflex_mood' + '?session=' + session, 'chart_reflex_mood', 'Reflex Mood', false)
    }


    loadSessionFiles();
    loadSessionData();
    setInterval(loadSessionData, 60000);

    $("#export").click(function() {
        var x = document.getElementById("session");
        var session = x.options[x.selectedIndex].text;
        window.location.href = "/plugins/session-stats/export?session=" + session;
    });
    });
{% endblock %}

{% block content %}
    <select id="session">
        <option selected>Current</option>
    </select>
    <input type="button" id="export" value="Export to JSON" />
    <div id="chart_os" class="chart"></div>
    <div id="chart_temp" class="chart"></div>
    <div id="chart_wifi" class="chart"></div>
    <div id="chart_duration" class="chart"></div>
    <div id="chart_reward" class="chart"></div>
    <div id="chart_epoch" class="chart"></div>
    <div id="chart_reflex_error" class="chart"></div>
    <div id="chart_reflex_sys" class="chart"></div>
    <div id="chart_reflex_mood" class="chart"></div>
{% endblock %}
"""

class GhettoClock:
    def __init__(self):
        self.lock = threading.Lock()
        self._track = datetime.now()
        self._counter_thread = threading.Thread(target=self.counter)
        self._counter_thread.daemon = True
        self._counter_thread.start()

    def counter(self):
        prctl.set_name("SessStatClock")
        while True:
            with self.lock:
                self._track += timedelta(seconds=1)
            sleep(1)

    def now(self):
        with self.lock:
            return self._track


class SessionStats(plugins.Plugin):
    __author__ = '33197631+dadav@users.noreply.github.com'
    __version__ = '0.1.0'
    __license__ = 'GPL3'
    __description__ = 'This plugin displays stats of the current session.'

    def __init__(self):
        self.lock = threading.Lock()
        self.options = dict()
        self.stats = dict()
        self.clock = GhettoClock()

    def on_loaded(self):
        """
        Gets called when the plugin gets loaded
        """
        # this has to happen in "loaded" because the options are not yet
        # available in the __init__
        os.makedirs(self.options['save_directory'], exist_ok=True)
        self.session_name = "stats_{}.json".format(self.clock.now().strftime("%Y_%m_%d_%H_%M"))
        self.session = StatusFile(os.path.join(self.options['save_directory'],
                                               self.session_name),
                                  data_format='json')
        logging.info("Session-stats plugin loaded.")

    def on_epoch(self, agent, epoch, epoch_data):
        """
        Save the epoch_data to self.stats
        """
        if hasattr(agent, '_reflex') and agent._reflex:
            epoch_data['timeout_errors'] = agent._reflex.state.get('timeout_errors', 0)
            epoch_data['injection_errors'] = agent._reflex.state.get('injection_errors', 0)
            epoch_data['io_wait'] = agent._reflex.state.get('io_wait', 0)
            epoch_data['is_promiscuous'] = agent._reflex.state.get('is_promiscuous', 0)
            epoch_data['stress'] = agent._reflex.stress_level
            epoch_data['risk'] = agent._reflex.risk
            epoch_data['bias'] = agent._reflex.bias()
        else:
            epoch_data['timeout_errors'] = 0
            epoch_data['injection_errors'] = 0
            epoch_data['io_wait'] = 0
            epoch_data['is_promiscuous'] = 0
            epoch_data['stress'] = 0
            epoch_data['risk'] = 0
            epoch_data['bias'] = {}

        with self.lock:
            self.stats[self.clock.now().strftime("%H:%M:%S")] = epoch_data
            self.session.update(data={'data': self.stats})

    @staticmethod
    def extract_key_values(data, subkeys):
        result = dict()
        result['values'] = list()
        result['labels'] = subkeys
        for plot_key in subkeys:
            v = [ [ts,d.get(plot_key, 0)] for ts, d in data.items()]
            result['values'].append(v)
        return result

    def on_webhook(self, path, request):
        if not path or path == "/":
            return render_template_string(TEMPLATE)

        session_param = request.args.get('session')

        if path == "os":
            extract_keys = ['cpu_load','mem_usage',]
        elif path == "temp":
            extract_keys = ['temperature']
        elif path == "wifi":
            extract_keys = [
                'missed_interactions',
                'num_hops',
                'num_peers',
                'tot_bond',
                'avg_bond',
                'num_deauths',
                'num_associations',
                'num_handshakes',
            ]
        elif path == "duration":
            extract_keys = [
                'duration_secs',
                'slept_for_secs',
            ]
        elif path == "reward":
            extract_keys = [
                'reward',
            ]
        elif path == "epoch":
            extract_keys = [
                'active_for_epochs',
            ]
        elif path == "reflex_error":
            extract_keys = [
                'timeout_errors',
                'injection_errors',
            ]
        elif path == "reflex_sys":
            extract_keys = [
                'io_wait',
                'is_promiscuous',
            ]
        elif path == "reflex_mood":
            extract_keys = [
                'stress',
                'risk',
            ]
        elif path == "session":
            return jsonify({'files': sorted(os.listdir(self.options['save_directory']), reverse=True)})
        elif path == "export":
            with self.lock:
                data = self.stats
                if session_param and session_param != 'Current':
                    file_stats = StatusFile(os.path.join(self.options['save_directory'], session_param), data_format='json')
                    data = file_stats.data_field_or('data', default=dict())
            response = jsonify(data)
            response.headers['Content-Disposition'] = 'attachment; filename=session_stats_{}.json'.format(session_param)
            return response

        with self.lock:
            data = self.stats
            if session_param and session_param != 'Current':
                file_stats = StatusFile(os.path.join(self.options['save_directory'], session_param), data_format='json')
                data = file_stats.data_field_or('data', default=dict())
            return jsonify(SessionStats.extract_key_values(data, extract_keys))
