"""
ids.capture — Real-time packet capture using Scapy
"""

import logging
from typing import Iterator

logger = logging.getLogger(__name__)


class PacketCapture:
    """
    Wraps Scapy sniff() in a clean iterator interface.

    Parameters
    ----------
    interface : network interface name (e.g. 'eth0', 'en0')
    """

    def __init__(self, interface: str) -> None:
        try:
            from scapy.all import conf
            conf.verb = 0  # suppress Scapy output
        except ImportError:
            raise ImportError(
                "Scapy is required for live capture.\n"
                "Install with: pip install scapy"
            )
        self.interface = interface
        self._running = False
        self._packets = []

    def stop(self) -> None:
        """Signal the capture loop to stop."""
        self._running = False

    def stream(self) -> Iterator:
        """
        Yield packets one by one as they arrive on the interface.

        Yields
        ------
        scapy Packet objects
        """
        from scapy.all import sniff

        self._running = True
        logger.info("Starting capture on %s", self.interface)

        def _on_packet(pkt):
            self._packets.append(pkt)

        sniff(
            iface=self.interface,
            prn=_on_packet,
            store=False,
            stop_filter=lambda _: not self._running,
        )

        yield from self._packets
