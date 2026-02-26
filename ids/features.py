"""
ids.features — Extract numeric features from raw Scapy packets
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Feature vector indices (must match training data column order)
FEATURE_NAMES = [
    "packet_size",       # Total length of the packet in bytes
    "is_tcp",            # 1 if TCP, 0 otherwise
    "is_udp",            # 1 if UDP, 0 otherwise
    "is_icmp",           # 1 if ICMP, 0 otherwise
    "src_port",          # Source port (0 for ICMP)
    "dst_port",          # Destination port (0 for ICMP)
    "tcp_flags_syn",     # SYN flag set
    "tcp_flags_rst",     # RST flag set
    "tcp_flags_fin",     # FIN flag set
    "tcp_flags_ack",     # ACK flag set
    "payload_size",      # Size of application payload
    "ttl",               # Time-to-live value
    "ip_frag",           # Fragmentation flag
    "header_length",     # IP header length
]


class FeatureExtractor:
    """
    Converts a Scapy packet into a fixed-length numpy feature vector.
    """

    def extract(self, packet) -> Optional[np.ndarray]:
        """
        Extract features from a packet.

        Parameters
        ----------
        packet : Scapy packet

        Returns
        -------
        numpy array of shape (14,) or None if packet is unsupported
        """
        try:
            return self._extract(packet)
        except Exception as exc:
            logger.debug("Feature extraction failed: %s", exc)
            return None

    def _extract(self, packet) -> Optional[np.ndarray]:
        try:
            from scapy.layers.inet import IP, TCP, UDP, ICMP
        except ImportError:
            return None

        if not packet.haslayer(IP):
            return None

        ip = packet[IP]
        is_tcp  = int(packet.haslayer(TCP))
        is_udp  = int(packet.haslayer(UDP))
        is_icmp = int(packet.haslayer(ICMP))

        src_port = dst_port = 0
        tcp_syn = tcp_rst = tcp_fin = tcp_ack = 0

        if is_tcp:
            tcp = packet[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            flags = tcp.flags
            tcp_syn = int(bool(flags & 0x02))
            tcp_rst = int(bool(flags & 0x04))
            tcp_fin = int(bool(flags & 0x01))
            tcp_ack = int(bool(flags & 0x10))
        elif is_udp:
            udp = packet["UDP"]
            src_port = udp.sport
            dst_port = udp.dport

        payload_size = len(bytes(packet.payload.payload)) if packet.payload else 0
        packet_size  = len(packet)
        ttl          = ip.ttl
        ip_frag      = int(ip.flags.MF or ip.frag > 0)
        header_len   = ip.ihl * 4 if ip.ihl else 20

        return np.array([
            packet_size, is_tcp, is_udp, is_icmp,
            src_port, dst_port,
            tcp_syn, tcp_rst, tcp_fin, tcp_ack,
            payload_size, ttl, ip_frag, header_len,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list:
        return FEATURE_NAMES
