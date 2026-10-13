"""Seed initial catalog data: agents, tools, demo data."""
import datetime
from database import SessionLocal, Base, engine
from models import (
    AgentConfiguration, ModelConfiguration, ToolConfiguration,
    Assessment, Finding, Workflow, Report, Conversation
)

CATALOG_AGENTS = [
    {
        "name": "PentestGPT Legacy",
        "description": "An LLM-based conversational penetration-testing assistant that helps plan and reason through assessment steps. Requires an OpenAI-compatible API key.",
        "adapter_type": "openai_api",
        "endpoint": "",
        "capabilities": ["planning", "reasoning", "reconnaissance", "exploitation_guidance"],
        "supported_models": ["gpt-4", "gpt-3.5-turbo", "llama3.2"],
        "runtime_requirements": "OpenAI-compatible API endpoint and key",
        "status": "not_configured",
    },
    {
        "name": "HackingBuddyGPT",
        "description": "An autonomous LLM-based hacking agent for security research. Requires an LLM API endpoint and configured scope.",
        "adapter_type": "openai_api",
        "endpoint": "",
        "capabilities": ["autonomous_recon", "port_scanning", "web_exploitation"],
        "supported_models": ["gpt-4", "gpt-3.5-turbo"],
        "runtime_requirements": "OpenAI-compatible API endpoint and key",
        "status": "not_configured",
    },
    {
        "name": "CAI",
        "description": "Cybersecurity AI (CAI) — a framework for automated penetration testing using LLMs. Supports various tools and scanning capabilities.",
        "adapter_type": "custom",
        "endpoint": "",
        "capabilities": ["reconnaissance", "vulnerability_scanning", "web_testing"],
        "supported_models": ["gpt-4", "llama3.2"],
        "runtime_requirements": "Python runtime, LLM API access, tool dependencies",
        "status": "not_configured",
    },
    {
        "name": "HexStrike AI",
        "description": "An AI-driven security testing framework that combines LLM reasoning with security tool orchestration.",
        "adapter_type": "custom",
        "endpoint": "",
        "capabilities": ["reconnaissance", "vulnerability_scanning", "exploitation"],
        "supported_models": ["gpt-4", "claude-3"],
        "runtime_requirements": "LLM API access, external runner",
        "status": "not_configured",
    },
    {
        "name": "PentAGI",
        "description": "A pentesting AGI framework with multi-agent orchestration for comprehensive security assessments.",
        "adapter_type": "custom",
        "endpoint": "",
        "capabilities": ["multi_agent", "reconnaissance", "exploitation", "reporting"],
        "supported_models": ["gpt-4", "llama3.2"],
        "runtime_requirements": "Docker, LLM API access",
        "status": "not_configured",
    },
    {
        "name": "SwarmAttacker",
        "description": "A swarm-based multi-agent system for coordinated penetration testing with distributed task execution.",
        "adapter_type": "custom",
        "endpoint": "",
        "capabilities": ["distributed_scan", "coordinated_attack", "reconnaissance"],
        "supported_models": ["gpt-4", "llama3.2"],
        "runtime_requirements": "Docker, LLM API access, network access to targets",
        "status": "not_configured",
    },
    {
        "name": "Master Security Agent",
        "description": "The central orchestration agent of SENTINEL NEXUS AI. Coordinates other agents, models, and tools for unified assessment workflows.",
        "adapter_type": "orchestrator",
        "endpoint": "internal",
        "capabilities": ["orchestration", "planning", "scope_validation", "finding_normalization", "reporting"],
        "supported_models": ["all"],
        "runtime_requirements": "Internal — activated when at least one model is configured",
        "status": "configured",
    },
]

CATALOG_TOOLS = [
    {"name": "Nmap", "category": "recon", "description": "Network discovery and security auditing utility for port scanning and service detection.", "task_types": ["port_scan", "service_detection", "os_detection"], "runtime_requirements": "External runner with nmap installed"},
    {"name": "OWASP ZAP", "category": "web", "description": "Web application security scanner for finding vulnerabilities in web applications.", "task_types": ["web_scan", "spider", "active_scan"], "runtime_requirements": "External runner with ZAP installed"},
    {"name": "Nuclei", "category": "vuln", "description": "Fast, customizable vulnerability scanner based on templates.", "task_types": ["vuln_scan", "template_scan"], "runtime_requirements": "External runner with nuclei installed"},
    {"name": "Nikto", "category": "web", "description": "Web server scanner for dangerous files and outdated software.", "task_types": ["web_scan", "server_check"], "runtime_requirements": "External runner with nikto installed"},
    {"name": "WhatWeb", "category": "recon", "description": "Identifies websites — CMS, blogging platforms, analytics packages, JavaScript libraries.", "task_types": ["web_fingerprint"], "runtime_requirements": "External runner with whatweb installed"},
    {"name": "testssl.sh", "category": "crypto", "description": "Tests TLS/SSL encryption of any service that supports TLS.", "task_types": ["ssl_scan", "tls_check"], "runtime_requirements": "External runner with testssl.sh installed"},
    {"name": "Semgrep", "category": "sast", "description": "Static analysis tool for finding bugs and enforcing code standards.", "task_types": ["static_analysis", "code_scan"], "runtime_requirements": "External runner with semgrep installed"},
    {"name": "Trivy", "category": "container", "description": "Comprehensive vulnerability scanner for containers and filesystems.", "task_types": ["container_scan", "fs_scan"], "runtime_requirements": "External runner with trivy installed"},
    {"name": "Checkov", "category": "iac", "description": "Infrastructure-as-code scanner for Terraform, CloudFormation, and more.", "task_types": ["iac_scan"], "runtime_requirements": "External runner with checkov installed"},
    {"name": "Gitleaks", "category": "secrets", "description": "Detects and prevents hardcoded secrets in source code.", "task_types": ["secret_scan"], "runtime_requirements": "External runner with gitleaks installed"},
    {"name": "SQLmap", "category": "exploit", "description": "Automatic SQL injection and database takeover tool.", "task_types": ["sqli_detection", "sqli_exploitation"], "runtime_requirements": "External runner with sqlmap installed — requires explicit authorization"},
]

CATALOG_MODELS = [
    {"display_name": "Ollama (llama3.2)", "provider": "ollama", "model_identifier": "llama3.2", "endpoint": "http://localhost:11434", "capabilities": ["chat", "completion"], "notes": "Suggested local model. Requires Ollama running and accessible from the backend."},
]

DEMO_WORKFLOWS = [
    {
        "name": "Standard Web Assessment",
        "description": "A standard workflow for authorized web application security assessments.",
        "steps": [
            {"id": "s1", "name": "Start Assessment", "type": "start", "status": "pending", "config": {}},
            {"id": "s2", "name": "Validate Scope", "type": "validate_scope", "status": "pending", "config": {}},
            {"id": "s3", "name": "Check Integration", "type": "check_integration", "status": "pending", "config": {}},
            {"id": "s4", "name": "Select Agent", "type": "select_agent", "status": "pending", "config": {}},
            {"id": "s5", "name": "Run Nmap Recon", "type": "run_tool", "status": "pending", "config": {"tool": "Nmap"}},
            {"id": "s6", "name": "Run Nuclei Scan", "type": "run_tool", "status": "pending", "config": {"tool": "Nuclei"}},
            {"id": "s7", "name": "Collect Output", "type": "collect_output", "status": "pending", "config": {}},
            {"id": "s8", "name": "Analyze Results", "type": "analyze", "status": "pending", "config": {}},
            {"id": "s9", "name": "Normalize Findings", "type": "normalize", "status": "pending", "config": {}},
            {"id": "s10", "name": "Request Approval", "type": "request_approval", "status": "pending", "config": {}},
            {"id": "s11", "name": "Generate Report", "type": "generate_report", "status": "pending", "config": {}},
            {"id": "s12", "name": "Complete", "type": "complete", "status": "pending", "config": {}},
        ],
    },
    {
        "name": "Passive Reconnaissance",
        "description": "Non-intrusive information gathering workflow. Safe for initial assessment phases.",
        "steps": [
            {"id": "s1", "name": "Start Assessment", "type": "start", "status": "pending", "config": {}},
            {"id": "s2", "name": "Validate Scope", "type": "validate_scope", "status": "pending", "config": {}},
            {"id": "s3", "name": "Select Agent", "type": "select_agent", "status": "pending", "config": {}},
            {"id": "s4", "name": "Run WhatWeb", "type": "run_tool", "status": "pending", "config": {"tool": "WhatWeb"}},
            {"id": "s5", "name": "Run testssl.sh", "type": "run_tool", "status": "pending", "config": {"tool": "testssl.sh"}},
            {"id": "s6", "name": "Collect Output", "type": "collect_output", "status": "pending", "config": {}},
            {"id": "s7", "name": "Analyze Results", "type": "analyze", "status": "pending", "config": {}},
            {"id": "s8", "name": "Normalize Findings", "type": "normalize", "status": "pending", "config": {}},
            {"id": "s9", "name": "Generate Report", "type": "generate_report", "status": "pending", "config": {}},
            {"id": "s10", "name": "Complete", "type": "complete", "status": "pending", "config": {}},
        ],
    },
]


def seed_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Seed agents
        if db.query(AgentConfiguration).count() == 0:
            for a in CATALOG_AGENTS:
                db.add(AgentConfiguration(**a, is_custom=False))
            db.commit()

        # Seed models
        if db.query(ModelConfiguration).count() == 0:
            for m in CATALOG_MODELS:
                db.add(ModelConfiguration(**m))
            db.commit()

        # Seed tools
        if db.query(ToolConfiguration).count() == 0:
            for t in CATALOG_TOOLS:
                db.add(ToolConfiguration(**t, is_custom=False, status="not_configured"))
            db.commit()

        # Seed workflows
        if db.query(Workflow).count() == 0:
            for w in DEMO_WORKFLOWS:
                db.add(Workflow(**w))
            db.commit()

        # Seed demo assessment + findings
        if db.query(Assessment).count() == 0:
            demo = Assessment(
                name="[DEMO] Juice Shop Assessment",
                description="Demonstration assessment against OWASP Juice Shop for thesis presentation.",
                target_type="web_app",
                target_identifier="http://localhost:3000",
                scope=["localhost", "127.0.0.1"],
                environment_type="juice_shop",
                selected_agents=["Master Security Agent"],
                selected_tools=["Nmap", "Nuclei", "OWASP ZAP"],
                objective="Passive and active vulnerability assessment of OWASP Juice Shop.",
                allowed_operations=["reconnaissance", "vulnerability_scanning"],
                authorization_confirmed=True,
                status="completed",
                progress=100.0,
                is_demo=True,
            )
            db.add(demo)
            db.commit()

            demo_findings = [
                {"title": "[DEMO] Outdated TLS Configuration", "severity": "medium", "confidence": "high", "source_tool": "testssl.sh", "target": "localhost:443", "description": "Server supports TLS 1.0 and TLS 1.1 which are deprecated.", "evidence": "testssl.sh output: TLS 1.0 offered (deprecated)", "remediation": "Disable TLS 1.0 and 1.1, enforce TLS 1.2+.", "validation_status": "confirmed", "review_status": "confirmed"},
                {"title": "[DEMO] SQL Injection in Login Form", "severity": "critical", "confidence": "high", "source_tool": "OWASP ZAP", "target": "localhost:3000/login", "description": "SQL injection vulnerability detected in the login endpoint.", "evidence": "ZAP alert: SQL Injection - payload ' OR 1=1 -- returned valid session.", "remediation": "Use parameterized queries and input validation.", "validation_status": "confirmed", "review_status": "confirmed"},
                {"title": "[DEMO] Missing Security Headers", "severity": "low", "confidence": "medium", "source_tool": "Nuclei", "target": "localhost:3000", "description": "Several security headers are missing from HTTP responses.", "evidence": "Missing: X-Content-Type-Options, X-Frame-Options, Content-Security-Policy", "remediation": "Add standard security headers to web server configuration.", "validation_status": "new", "review_status": "new"},
                {"title": "[DEMO] Open Port 8080 Detected", "severity": "informational", "confidence": "high", "source_tool": "Nmap", "target": "localhost:8080", "description": "Port 8080 is open and running an unidentified service.", "evidence": "Nmap: 8080/tcp open http-proxy", "remediation": "Verify if this port should be exposed and restrict access if not needed.", "validation_status": "new", "review_status": "new"},
            ]
            for f in demo_findings:
                db.add(Finding(**f, assessment_id=demo.id, is_demo=True))
            db.commit()

            # Demo report
            db.add(Report(
                assessment_id=demo.id,
                title="[DEMO] Juice Shop Assessment Report",
                status="generated",
                is_demo=True,
                content={
                    "executive_summary": "This demonstration report covers the OWASP Juice Shop assessment.",
                    "methodology": "Reconnaissance, vulnerability scanning, and manual verification.",
                    "tools_used": ["Nmap", "Nuclei", "OWASP ZAP", "testssl.sh"],
                }
            ))
            db.commit()

            # Demo conversation
            db.add(Conversation(
                title="[DEMO] Juice Shop Assessment Chat",
                assessment_id=demo.id,
            ))
            db.commit()
    finally:
        db.close()
