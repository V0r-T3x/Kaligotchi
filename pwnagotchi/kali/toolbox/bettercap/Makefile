.ONESHELL:

# Configuration
BETTERCAP_VERSION ?= v2.41.4
CAPLETS_REPO      ?= https://github.com/v0r-T3x/caplets.git
TMPDIR            ?= /var/tmp
GO_CACHE          ?= $(TMPDIR)/go-cache

# Dependencies
DEPS := build-essential libpcap-dev libusb-1.0-0-dev libnetfilter-queue-dev golang-go git wget

.PHONY: all install uninstall install-deps setup-libpcap-compat build-bettercap install-caplets clean install-service

all: install

install: install-deps setup-libpcap-compat build-bettercap install-caplets install-service
	@echo "Bettercap installation complete."

install-deps:
	@echo "--> Updating package lists and installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y $(DEPS)

setup-libpcap-compat:
	@echo "--> Compiling and installing libpcap 1.9.1 to fix RPi monitor mode bug in 1.10.x..."
	@if [ ! -f /usr/local/lib/libpcap.so.1.9.1 ]; then \
		cd $(TMPDIR) && \
		wget -q --show-progress https://www.tcpdump.org/release/libpcap-1.9.1.tar.gz && \
		tar xzf libpcap-1.9.1.tar.gz && \
		cd $(TMPDIR)/libpcap-1.9.1 && \
		./configure --prefix=/usr/local && \
		make -j1 && \
		sudo make install && \
		sudo ln -sf /usr/local/lib/libpcap.so.1.9.1 /usr/local/lib/libpcap.so.1 && \
		sudo ldconfig && \
		cd / && sudo rm -rf $(TMPDIR)/libpcap-1.9.1 $(TMPDIR)/libpcap-1.9.1.tar.gz && \
		echo "libpcap 1.9.1 installed successfully."; \
	else \
		echo "    libpcap 1.9.1 already installed. Skipping."; \
	fi;

build-bettercap:
	@echo "--> Installing bettercap $(BETTERCAP_VERSION)..."
	# Remove old APT packages if present
	@if dpkg -l | grep -q '^ii.*bettercap'; then \
		echo "    Removing old APT bettercap packages..."; \
		sudo apt remove -y bettercap bettercap-caplets; \
		sudo apt autoremove -y; \
	fi

	# Check if already installed (checking for v2.41.x to match original logic)
	@if [ -f /usr/local/bin/bettercap ] && /usr/local/bin/bettercap --version 2>/dev/null | grep -q "v2\.41\."; then \
		echo "    bettercap v2.41.x already installed – skipping build."; \
	else \
		echo "    Building bettercap from source..."; \
		echo "    Creating temporary 4GB swap..."; \
		sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; \
		sudo swapoff /swapfile-temp-4g 2>/dev/null || true; \
		sudo rm -f /swapfile-temp-4g; \
		sudo fallocate -l 4G /swapfile-temp-4g; \
		sudo chmod 600 /swapfile-temp-4g; \
		sudo mkswap /swapfile-temp-4g; \
		sudo swapon /swapfile-temp-4g; \
		\
		echo "    Compiling bettercap $(BETTERCAP_VERSION) with ARM fix..."; \
		sudo bash -c '\
			set -e; \
			export PATH=/usr/local/bin:$$PATH; \
			export GOMAXPROCS=1; \
			export TMPDIR=$(TMPDIR); \
			export GOCACHE=$(GO_CACHE); \
			mkdir -p $(GO_CACHE); \
			cd $(TMPDIR); \
			rm -rf bettercap gopacket-patched; \
			git clone --depth 1 https://github.com/bettercap/bettercap.git; \
			cd bettercap; \
			git fetch --depth=1 origin tag $(BETTERCAP_VERSION); \
			git checkout $(BETTERCAP_VERSION); \
			git submodule update --init --recursive; \
			go mod download; \
			ARCH=$$(uname -m); \
			if [ "$$ARCH" = "aarch64" ] || [ "$$ARCH" = "x86_64" ]; then \
				echo "    64-bit system ($$ARCH) detected. Skipping 32-bit time_t patch."; \
			else \
				echo "    32-bit system ($$ARCH) detected. Applying gopacket time_t patch..."; \
				GOPACKET_DIR=$$(go list -m -mod=mod -f '\''{{.Dir}}'\'' github.com/google/gopacket); \
				PATCHDIR=$(TMPDIR)/gopacket-patched; \
				cp -r "$$GOPACKET_DIR" "$$PATCHDIR"; \
				sed -i "s/C.gopacket_time_secs_t/C.__time64_t/g"   "$$PATCHDIR/pcap/pcap_unix.go"; \
				sed -i "s/C.gopacket_time_usecs_t/C.__suseconds64_t/g" "$$PATCHDIR/pcap/pcap_unix.go"; \
				echo "replace github.com/google/gopacket => $$PATCHDIR" >> go.mod; \
				go mod tidy; \
			fi; \
			go build -gcflags="all=-N -l" -ldflags="-s -w" -o bettercap .; \
			mv bettercap /usr/local/bin/bettercap; \
			echo "    bettercap $(BETTERCAP_VERSION) successfully installed!"; \
		'; \
		\
		echo "    Cleaning up temporary files..."; \
		sudo swapoff /swapfile-temp-4g 2>/dev/null || true; \
		sudo rm -f /swapfile-temp-4g; \
		sudo rm -rf $(TMPDIR)/gopacket-patched $(GO_CACHE) $(TMPDIR)/bettercap; \
		sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; \
	fi

install-caplets:
	@echo "--> Installing caplets..."
	@cd $(TMPDIR) && \
	rm -rf caplets && \
	git clone --depth 1 $(CAPLETS_REPO) && \
	cd caplets && \
	sudo make install > /dev/null 2>&1; \
	echo "    Latest caplets installed."
	@echo "--> Patching http-ui caplet for Pwnagotchi..."
	sudo sed -i 's/set api.rest.username user/set api.rest.username pwnagotchi/' /usr/local/share/bettercap/caplets/http-ui.cap
	sudo sed -i 's/set api.rest.password pass/set api.rest.password pwnagotchi/' /usr/local/share/bettercap/caplets/http-ui.cap
	@echo "    http-ui caplet patched successfully."

install-service:
	@echo "--> Installing bettercap systemd service and launcher..."
	sudo cp -v data/usr/bin/bettercap-launcher /usr/bin/
	sudo chmod 755 /usr/bin/bettercap-launcher
	sudo cp -v data/etc/systemd/system/bettercap.service /etc/systemd/system/
	sudo chmod 644 /etc/systemd/system/bettercap.service
	@echo "--> Reloading systemd daemon and enabling bettercap service..."
	sudo systemctl daemon-reload
	sudo systemctl enable bettercap.service
	@echo "    Bettercap service installed and enabled."

uninstall:
	@echo "--> Uninstalling bettercap and its service..."
	-sudo systemctl stop bettercap.service
	-sudo systemctl disable bettercap.service
	sudo rm -f /usr/local/bin/bettercap
	sudo rm -rf /usr/local/share/bettercap
	sudo rm -f /usr/bin/bettercap-launcher /etc/systemd/system/bettercap.service
	-sudo systemctl daemon-reload
	@echo "    Bettercap and its service have been uninstalled."

clean:
	sudo rm -rf $(TMPDIR)/bettercap $(TMPDIR)/gopacket-patched $(TMPDIR)/caplets $(GO_CACHE)