"""
AML Fraud Detection Data & Engine Layer
Provides realistic transaction graph datasets, customer profiles, risk metrics,
and explainable AI (XAI) pattern explanations.
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

# Master Transaction Registry
TRANSACTIONS = [
    {
        "tx_id": "TX-10231",
        "account": "800A89B40",
        "amount": 820000,
        "amount_formatted": "8,20,000",
        "risk": "High",
        "risk_score": 91,
        "is_fraud": True,
        "pattern": "Fan-Out",
        "timestamp": "05 May 2025, 12:32:10 PM",
        "to_accounts_str": "5 Accounts",
        "to_accounts": ["ACCT-801", "ACCT-802", "ACCT-803", "ACCT-804", "ACCT-805"],
        "payment_format": "NEFT",
        "model_used": "GraphSAGE + XGBoost Ensemble",
        "model_confidence": "91%",
        "explanations": [
            "Fan-Out pattern detected: 1 sender → 5 new receivers within 18 minutes",
            "89% of receiver accounts created within the last 7 days",
            "Transaction velocity 4.8× higher than 30-day baseline",
            "Amount ₹8,20,000 is 4.2 IQR above account's historical median (₹35,000)",
            "Multiple NEFT transfers executed in burst pattern — typical layering indicator",
            "Network centrality score of source node elevated to 0.94 (threshold: 0.70)"
        ],
        "fan_out_sub_transactions": [
            {"sub_tx_id": "TX-10231-A", "to_account": "ACCT-801", "amount": "1,64,000", "time": "12:32:10 PM", "account_age_days": 3, "status": "Completed"},
            {"sub_tx_id": "TX-10231-B", "to_account": "ACCT-802", "amount": "1,64,000", "time": "12:34:22 PM", "account_age_days": 5, "status": "Completed"},
            {"sub_tx_id": "TX-10231-C", "to_account": "ACCT-803", "amount": "1,64,000", "time": "12:36:48 PM", "account_age_days": 2, "status": "Completed"},
            {"sub_tx_id": "TX-10231-D", "to_account": "ACCT-804", "amount": "1,64,000", "time": "12:41:05 PM", "account_age_days": 8, "status": "Completed"},
            {"sub_tx_id": "TX-10231-E", "to_account": "ACCT-805", "amount": "1,64,000", "time": "12:50:30 PM", "account_age_days": 1, "status": "Completed"},
        ]
    },
    {
        "tx_id": "TX-10228",
        "account": "800F22A10",
        "amount": 1450000,
        "amount_formatted": "14,50,000",
        "risk": "High",
        "risk_score": 88,
        "is_fraud": True,
        "pattern": "Circular Money Flow",
        "timestamp": "05 May 2025, 11:45:00 AM",
        "to_accounts_str": "3 Accounts (Ring)",
        "to_accounts": ["ACCT-901", "ACCT-902", "800F22A10"],
        "payment_format": "RTGS",
        "model_used": "GNN GraphSAGE + Isolation Forest",
        "model_confidence": "88%",
        "explanations": [
            "Circular transfer loop: 800F22A10 → ACCT-901 → ACCT-902 → 800F22A10",
            "Full round-trip completed in under 120 seconds",
            "Final received amount matches origin within 98.5% — near-zero leakage",
            "Zero commercial rationale: no invoice or goods transfer associated",
            "GNN structural embedding flagged this sub-graph as a known laundering motif",
            "All ring accounts share the same registered mobile number"
        ],
        "fan_out_sub_transactions": [
            {"sub_tx_id": "TX-10228-A", "to_account": "ACCT-901", "amount": "14,50,000", "time": "11:45:00 AM", "account_age_days": 14, "status": "Completed"},
            {"sub_tx_id": "TX-10228-B", "to_account": "ACCT-902", "amount": "14,48,000", "time": "11:46:15 AM", "account_age_days": 14, "status": "Completed"},
            {"sub_tx_id": "TX-10228-C", "to_account": "800F22A10", "amount": "14,42,700", "time": "11:47:02 AM", "account_age_days": 1240, "status": "Completed"},
        ]
    },
    {
        "tx_id": "TX-10221",
        "account": "800C91B20",
        "amount": 210000,
        "amount_formatted": "2,10,000",
        "risk": "Medium",
        "risk_score": 64,
        "is_fraud": False,
        "pattern": "Rapid Velocity",
        "timestamp": "05 May 2025, 10:15:22 AM",
        "to_accounts_str": "2 Accounts",
        "to_accounts": ["ACCT-701", "ACCT-702"],
        "payment_format": "IMPS",
        "model_used": "XGBoost Classifier",
        "model_confidence": "64%",
        "explanations": [
            "Transaction frequency spike in 1-hour window (3× standard deviation above baseline)",
            "Moderate deviation from 30-day baseline amount (+75%)",
            "New IP address region detected — geolocation mismatch",
            "Receiver accounts created within last 30 days"
        ],
        "fan_out_sub_transactions": [
            {"sub_tx_id": "TX-10221-A", "to_account": "ACCT-701", "amount": "1,05,000", "time": "10:15:22 AM", "account_age_days": 22, "status": "Completed"},
            {"sub_tx_id": "TX-10221-B", "to_account": "ACCT-702", "amount": "1,05,000", "time": "10:16:44 AM", "account_age_days": 28, "status": "Completed"},
        ]
    },
    {
        "tx_id": "TX-10218",
        "account": "800D11E30",
        "amount": 75000,
        "amount_formatted": "75,000",
        "risk": "Medium",
        "risk_score": 52,
        "is_fraud": False,
        "pattern": "Structuring / Smurfing",
        "timestamp": "05 May 2025, 09:30:45 AM",
        "to_accounts_str": "4 Accounts",
        "to_accounts": ["ACCT-601", "ACCT-602", "ACCT-603", "ACCT-604"],
        "payment_format": "UPI",
        "model_used": "Random Forest + Graph Analysis",
        "model_confidence": "52%",
        "explanations": [
            "Multiple consecutive transfers just below ₹1,00,000 mandatory reporting threshold",
            "Short time interval between transfers (avg 4 minutes)",
            "Account history shows steady low-volume balance — sudden burst suspicious"
        ],
        "fan_out_sub_transactions": [
            {"sub_tx_id": "TX-10218-A", "to_account": "ACCT-601", "amount": "19,000", "time": "09:30:45 AM", "account_age_days": 45, "status": "Completed"},
            {"sub_tx_id": "TX-10218-B", "to_account": "ACCT-602", "amount": "19,500", "time": "09:34:12 AM", "account_age_days": 60, "status": "Completed"},
            {"sub_tx_id": "TX-10218-C", "to_account": "ACCT-603", "amount": "18,800", "time": "09:38:55 AM", "account_age_days": 30, "status": "Completed"},
            {"sub_tx_id": "TX-10218-D", "to_account": "ACCT-604", "amount": "17,700", "time": "09:42:30 AM", "account_age_days": 90, "status": "Completed"},
        ]
    },
    {
        "tx_id": "TX-10215",
        "account": "800B77A20",
        "amount": 12500,
        "amount_formatted": "12,500",
        "risk": "Low",
        "risk_score": 12,
        "is_fraud": False,
        "pattern": "Normal Behavior",
        "timestamp": "05 May 2025, 08:12:00 AM",
        "to_accounts_str": "1 Account",
        "to_accounts": ["ACCT-501"],
        "payment_format": "UPI",
        "model_used": "Baseline Logistic Regression",
        "model_confidence": "98%",
        "explanations": [
            "Transaction consistent with 6-month historical spending pattern",
            "Known recurring beneficiary account",
            "Normal velocity and geolocation match"
        ],
        "fan_out_sub_transactions": [
            {"sub_tx_id": "TX-10215-A", "to_account": "ACCT-501", "amount": "12,500", "time": "08:12:00 AM", "account_age_days": 720, "status": "Completed"},
        ]
    },
    {
        "tx_id": "TX-10210",
        "account": "800E55C11",
        "amount": 540000,
        "amount_formatted": "5,40,000",
        "risk": "High",
        "risk_score": 85,
        "is_fraud": True,
        "pattern": "Account Takeover",
        "timestamp": "04 May 2025, 11:20:15 PM",
        "to_accounts_str": "3 Accounts",
        "to_accounts": ["ACCT-401", "ACCT-402", "ACCT-403"],
        "payment_format": "NEFT",
        "model_used": "GraphSAGE + Isolation Forest",
        "model_confidence": "85%",
        "explanations": [
            "Dormant account (inactive 180 days) reactivated with immediate high-value transfer",
            "Password reset performed 10 minutes prior to transaction",
            "Unrecognized device fingerprint and geolocation jump (Mumbai → Kolkata)"
        ],
        "fan_out_sub_transactions": [
            {"sub_tx_id": "TX-10210-A", "to_account": "ACCT-401", "amount": "1,80,000", "time": "11:20:15 PM", "account_age_days": 6, "status": "Completed"},
            {"sub_tx_id": "TX-10210-B", "to_account": "ACCT-402", "amount": "1,80,000", "time": "11:22:40 PM", "account_age_days": 4, "status": "Completed"},
            {"sub_tx_id": "TX-10210-C", "to_account": "ACCT-403", "amount": "1,80,000", "time": "11:24:08 PM", "account_age_days": 9, "status": "Completed"},
        ]
    },
    {
        "tx_id": "TX-10204",
        "account": "800G88D99",
        "amount": 95000,
        "amount_formatted": "95,000",
        "risk": "Low",
        "risk_score": 18,
        "is_fraud": False,
        "pattern": "Normal Behavior",
        "timestamp": "04 May 2025, 06:14:00 PM",
        "to_accounts_str": "1 Account",
        "to_accounts": ["ACCT-301"],
        "payment_format": "IMPS",
        "model_used": "XGBoost Classifier",
        "model_confidence": "95%",
        "explanations": [
            "Transaction matches regular vendor payment schedule",
            "Verified merchant recipient"
        ],
        "fan_out_sub_transactions": [
            {"sub_tx_id": "TX-10204-A", "to_account": "ACCT-301", "amount": "95,000", "time": "06:14:00 PM", "account_age_days": 1825, "status": "Completed"},
        ]
    }
]

# Extended Customer Profiles — sender accounts
CUSTOMER_PROFILES = {
    "800A89B40": {
        "account_id": "800A89B40",
        "name": "Rajesh Kumar Sharma",
        "kyc_status": "Verified",
        "account_type": "Current",
        "city": "Mumbai",
        "risk_tier": "High Risk",
        "open_since": "Jan 2019",
        "last_login": "05 May 2025, 12:28 PM",
        "device_count": 4,
        "total_transactions": "1,284",
        "total_incoming": "₹2.31 Cr",
        "total_outgoing": "₹2.48 Cr",
        "unique_senders": 18,
        "unique_receivers": 72,
        "avg_tx_amount": "₹35,000",
        "behavior_summary": [
            {"metric": "Transactions", "historical": "5", "current": "11", "change": "↑ 120%", "change_type": "high"},
            {"metric": "Unique Receivers", "historical": "1 - 3", "current": "9", "change": "↑ 300%", "change_type": "high"},
            {"metric": "New Receiver Ratio", "historical": "5 - 15%", "current": "89%", "change": "↑ 74pp", "change_type": "high"},
            {"metric": "Avg. Amount (₹)", "historical": "35,000", "current": "1,36,667", "change": "↑ 290%", "change_type": "high"},
        ]
    },
    "800F22A10": {
        "account_id": "800F22A10",
        "name": "Priya Venkataraman",
        "kyc_status": "Verified",
        "account_type": "Savings",
        "city": "Chennai",
        "risk_tier": "High Risk",
        "open_since": "Mar 2017",
        "last_login": "05 May 2025, 11:41 AM",
        "device_count": 2,
        "total_transactions": "2,450",
        "total_incoming": "₹5.10 Cr",
        "total_outgoing": "₹5.05 Cr",
        "unique_senders": 42,
        "unique_receivers": 38,
        "avg_tx_amount": "₹1,20,000",
        "behavior_summary": [
            {"metric": "Transactions", "historical": "8", "current": "24", "change": "↑ 200%", "change_type": "high"},
            {"metric": "Unique Receivers", "historical": "2 - 4", "current": "12", "change": "↑ 500%", "change_type": "high"},
            {"metric": "New Receiver Ratio", "historical": "10%", "current": "95%", "change": "↑ 85pp", "change_type": "high"},
            {"metric": "Avg. Amount (₹)", "historical": "1,20,000", "current": "4,83,333", "change": "↑ 302%", "change_type": "high"},
        ]
    },
    "800C91B20": {
        "account_id": "800C91B20",
        "name": "Amit Desai",
        "kyc_status": "Verified",
        "account_type": "Savings",
        "city": "Ahmedabad",
        "risk_tier": "Medium Risk",
        "open_since": "Jun 2020",
        "last_login": "05 May 2025, 10:10 AM",
        "device_count": 1,
        "total_transactions": "420",
        "total_incoming": "₹45.0 L",
        "total_outgoing": "₹42.5 L",
        "unique_senders": 12,
        "unique_receivers": 15,
        "avg_tx_amount": "₹40,000",
        "behavior_summary": [
            {"metric": "Transactions", "historical": "3", "current": "7", "change": "↑ 133%", "change_type": "medium"},
            {"metric": "Unique Receivers", "historical": "1 - 2", "current": "4", "change": "↑ 100%", "change_type": "medium"},
            {"metric": "New Receiver Ratio", "historical": "10%", "current": "40%", "change": "↑ 30pp", "change_type": "medium"},
            {"metric": "Avg. Amount (₹)", "historical": "40,000", "current": "70,000", "change": "↑ 75%", "change_type": "medium"},
        ]
    },
    "800D11E30": {
        "account_id": "800D11E30",
        "name": "Sunita Patel",
        "kyc_status": "Pending Review",
        "account_type": "Savings",
        "city": "Surat",
        "risk_tier": "Medium Risk",
        "open_since": "Sep 2021",
        "last_login": "05 May 2025, 09:25 AM",
        "device_count": 2,
        "total_transactions": "890",
        "total_incoming": "₹1.15 Cr",
        "total_outgoing": "₹1.10 Cr",
        "unique_senders": 25,
        "unique_receivers": 30,
        "avg_tx_amount": "₹25,000",
        "behavior_summary": [
            {"metric": "Transactions", "historical": "4", "current": "8", "change": "↑ 100%", "change_type": "medium"},
            {"metric": "Unique Receivers", "historical": "2", "current": "5", "change": "↑ 150%", "change_type": "medium"},
            {"metric": "New Receiver Ratio", "historical": "12%", "current": "50%", "change": "↑ 38pp", "change_type": "medium"},
            {"metric": "Avg. Amount (₹)", "historical": "25,000", "current": "37,500", "change": "↑ 50%", "change_type": "medium"},
        ]
    },
    "800B77A20": {
        "account_id": "800B77A20",
        "name": "Kiran Mehta",
        "kyc_status": "Verified",
        "account_type": "Savings",
        "city": "Pune",
        "risk_tier": "Low Risk",
        "open_since": "Nov 2016",
        "last_login": "05 May 2025, 08:05 AM",
        "device_count": 1,
        "total_transactions": "310",
        "total_incoming": "₹18.5 L",
        "total_outgoing": "₹16.2 L",
        "unique_senders": 8,
        "unique_receivers": 11,
        "avg_tx_amount": "₹15,000",
        "behavior_summary": [
            {"metric": "Transactions", "historical": "2", "current": "2", "change": "0%", "change_type": "low"},
            {"metric": "Unique Receivers", "historical": "1", "current": "1", "change": "0%", "change_type": "low"},
            {"metric": "New Receiver Ratio", "historical": "5%", "current": "0%", "change": "-5pp", "change_type": "low"},
            {"metric": "Avg. Amount (₹)", "historical": "15,000", "current": "12,500", "change": "↓ 16.6%", "change_type": "low"},
        ]
    },
    "800E55C11": {
        "account_id": "800E55C11",
        "name": "Vikram Nair",
        "kyc_status": "Flagged",
        "account_type": "Current",
        "city": "Bengaluru",
        "risk_tier": "High Risk",
        "open_since": "Aug 2015",
        "last_login": "04 May 2025, 11:10 PM",
        "device_count": 6,
        "total_transactions": "3,100",
        "total_incoming": "₹8.20 Cr",
        "total_outgoing": "₹8.15 Cr",
        "unique_senders": 55,
        "unique_receivers": 89,
        "avg_tx_amount": "₹80,000",
        "behavior_summary": [
            {"metric": "Transactions", "historical": "6", "current": "18", "change": "↑ 200%", "change_type": "high"},
            {"metric": "Unique Receivers", "historical": "2 - 3", "current": "10", "change": "↑ 400%", "change_type": "high"},
            {"metric": "New Receiver Ratio", "historical": "8%", "current": "92%", "change": "↑ 84pp", "change_type": "high"},
            {"metric": "Avg. Amount (₹)", "historical": "80,000", "current": "1,80,000", "change": "↑ 125%", "change_type": "high"},
        ]
    },
    "800G88D99": {
        "account_id": "800G88D99",
        "name": "Deepa Krishnamurthy",
        "kyc_status": "Verified",
        "account_type": "Savings",
        "city": "Hyderabad",
        "risk_tier": "Low Risk",
        "open_since": "Feb 2018",
        "last_login": "04 May 2025, 06:08 PM",
        "device_count": 1,
        "total_transactions": "640",
        "total_incoming": "₹72.0 L",
        "total_outgoing": "₹68.5 L",
        "unique_senders": 10,
        "unique_receivers": 14,
        "avg_tx_amount": "₹55,000",
        "behavior_summary": [
            {"metric": "Transactions", "historical": "2", "current": "2", "change": "0%", "change_type": "low"},
            {"metric": "Unique Receivers", "historical": "1", "current": "1", "change": "0%", "change_type": "low"},
            {"metric": "New Receiver Ratio", "historical": "5%", "current": "0%", "change": "-5pp", "change_type": "low"},
            {"metric": "Avg. Amount (₹)", "historical": "55,000", "current": "95,000", "change": "↑ 72%", "change_type": "medium"},
        ]
    }
}

# Receiver Account Profiles (for right-panel on row click)
RECEIVER_PROFILES = {
    "ACCT-801": {"account_id": "ACCT-801", "name": "Rahul Gupta", "kyc_status": "Minimal KYC", "account_type": "Savings", "city": "Delhi", "risk_tier": "High Risk", "open_since": "02 May 2025", "device_count": 1, "total_transactions": "2", "total_incoming": "₹1,64,000", "total_outgoing": "₹1,63,500", "unique_senders": 1, "unique_receivers": 1, "avg_tx_amount": "₹1,64,000", "notes": "Newly opened account, immediate outflow detected"},
    "ACCT-802": {"account_id": "ACCT-802", "name": "Sneha Joshi", "kyc_status": "Minimal KYC", "account_type": "Savings", "city": "Noida", "risk_tier": "High Risk", "open_since": "30 Apr 2025", "device_count": 1, "total_transactions": "3", "total_incoming": "₹1,64,000", "total_outgoing": "₹1,63,800", "unique_senders": 1, "unique_receivers": 2, "avg_tx_amount": "₹82,000", "notes": "Same device fingerprint as ACCT-803"},
    "ACCT-803": {"account_id": "ACCT-803", "name": "Mohan Das", "kyc_status": "Minimal KYC", "account_type": "Savings", "city": "Gurgaon", "risk_tier": "High Risk", "open_since": "03 May 2025", "device_count": 1, "total_transactions": "1", "total_incoming": "₹1,64,000", "total_outgoing": "₹1,64,000", "unique_senders": 1, "unique_receivers": 1, "avg_tx_amount": "₹1,64,000", "notes": "Same device fingerprint as ACCT-802"},
    "ACCT-804": {"account_id": "ACCT-804", "name": "Kavita Singh", "kyc_status": "Minimal KYC", "account_type": "Savings", "city": "Faridabad", "risk_tier": "High Risk", "open_since": "27 Apr 2025", "device_count": 2, "total_transactions": "4", "total_incoming": "₹3,28,000", "total_outgoing": "₹3,25,000", "unique_senders": 2, "unique_receivers": 3, "avg_tx_amount": "₹82,000", "notes": "Receives from 2 different suspicious senders"},
    "ACCT-805": {"account_id": "ACCT-805", "name": "Arjun Tiwari", "kyc_status": "Minimal KYC", "account_type": "Savings", "city": "Delhi", "risk_tier": "High Risk", "open_since": "04 May 2025", "device_count": 1, "total_transactions": "1", "total_incoming": "₹1,64,000", "total_outgoing": "₹1,63,900", "unique_senders": 1, "unique_receivers": 1, "avg_tx_amount": "₹1,64,000", "notes": "Most recently opened receiver — just 1 day old"},
    "ACCT-901": {"account_id": "ACCT-901", "name": "Ring Node Alpha", "kyc_status": "Verified", "account_type": "Current", "city": "Chennai", "risk_tier": "High Risk", "open_since": "21 Apr 2025", "device_count": 2, "total_transactions": "12", "total_incoming": "₹14,50,000", "total_outgoing": "₹14,48,000", "unique_senders": 1, "unique_receivers": 1, "avg_tx_amount": "₹7,24,000", "notes": "Pass-through account in circular ring — no economic purpose"},
    "ACCT-902": {"account_id": "ACCT-902", "name": "Ring Node Beta", "kyc_status": "Verified", "account_type": "Current", "city": "Chennai", "risk_tier": "High Risk", "open_since": "21 Apr 2025", "device_count": 2, "total_transactions": "10", "total_incoming": "₹14,48,000", "total_outgoing": "₹14,42,700", "unique_senders": 1, "unique_receivers": 1, "avg_tx_amount": "₹7,21,000", "notes": "Shares registered mobile with ACCT-901 and source"},
    "ACCT-701": {"account_id": "ACCT-701", "name": "Meena Rajan", "kyc_status": "Verified", "account_type": "Savings", "city": "Coimbatore", "risk_tier": "Medium Risk", "open_since": "13 Apr 2025", "device_count": 1, "total_transactions": "8", "total_incoming": "₹4,20,000", "total_outgoing": "₹4,00,000", "unique_senders": 3, "unique_receivers": 4, "avg_tx_amount": "₹52,500", "notes": "Account 22 days old — moderate velocity"},
    "ACCT-702": {"account_id": "ACCT-702", "name": "Suresh Babu", "kyc_status": "Verified", "account_type": "Savings", "city": "Madurai", "risk_tier": "Medium Risk", "open_since": "07 Apr 2025", "device_count": 1, "total_transactions": "5", "total_incoming": "₹2,10,000", "total_outgoing": "₹2,05,000", "unique_senders": 2, "unique_receivers": 2, "avg_tx_amount": "₹42,000", "notes": "Account 28 days old"},
    "ACCT-601": {"account_id": "ACCT-601", "name": "Ramesh Yadav", "kyc_status": "Verified", "account_type": "Savings", "city": "Jaipur", "risk_tier": "Low Risk", "open_since": "20 Mar 2025", "device_count": 1, "total_transactions": "15", "total_incoming": "₹95,000", "total_outgoing": "₹90,000", "unique_senders": 4, "unique_receivers": 5, "avg_tx_amount": "₹6,333", "notes": "Established account — below-threshold transfers"},
    "ACCT-602": {"account_id": "ACCT-602", "name": "Geeta Sharma", "kyc_status": "Verified", "account_type": "Savings", "city": "Jaipur", "risk_tier": "Low Risk", "open_since": "04 Mar 2025", "device_count": 1, "total_transactions": "18", "total_incoming": "₹1,17,000", "total_outgoing": "₹1,10,000", "unique_senders": 5, "unique_receivers": 6, "avg_tx_amount": "₹6,500", "notes": "Normal spending pattern"},
    "ACCT-603": {"account_id": "ACCT-603", "name": "Dinesh Agarwal", "kyc_status": "Verified", "account_type": "Savings", "city": "Jodhpur", "risk_tier": "Low Risk", "open_since": "05 Apr 2025", "device_count": 2, "total_transactions": "10", "total_incoming": "₹75,200", "total_outgoing": "₹70,000", "unique_senders": 3, "unique_receivers": 4, "avg_tx_amount": "₹7,520", "notes": "Below threshold, consistent transfers"},
    "ACCT-604": {"account_id": "ACCT-604", "name": "Pooja Bansal", "kyc_status": "Verified", "account_type": "Savings", "city": "Udaipur", "risk_tier": "Low Risk", "open_since": "04 Feb 2025", "device_count": 1, "total_transactions": "22", "total_incoming": "₹1,59,300", "total_outgoing": "₹1,52,000", "unique_senders": 6, "unique_receivers": 8, "avg_tx_amount": "₹7,240", "notes": "Established account, normal behaviour"},
    "ACCT-501": {"account_id": "ACCT-501", "name": "LIC Premium Payment", "kyc_status": "Verified", "account_type": "Corporate", "city": "Mumbai", "risk_tier": "Low Risk", "open_since": "Jan 2010", "device_count": 0, "total_transactions": "42,000", "total_incoming": "₹85.0 Cr", "total_outgoing": "₹0", "unique_senders": 8500, "unique_receivers": 0, "avg_tx_amount": "₹20,238", "notes": "Verified LIC premium collection account"},
    "ACCT-401": {"account_id": "ACCT-401", "name": "Unknown Holder A", "kyc_status": "Minimal KYC", "account_type": "Savings", "city": "Kolkata", "risk_tier": "High Risk", "open_since": "26 Apr 2025", "device_count": 1, "total_transactions": "3", "total_incoming": "₹1,80,000", "total_outgoing": "₹1,79,500", "unique_senders": 1, "unique_receivers": 1, "avg_tx_amount": "₹60,000", "notes": "Geolocation matches suspicious takeover region"},
    "ACCT-402": {"account_id": "ACCT-402", "name": "Unknown Holder B", "kyc_status": "Minimal KYC", "account_type": "Savings", "city": "Kolkata", "risk_tier": "High Risk", "open_since": "26 Apr 2025", "device_count": 1, "total_transactions": "2", "total_incoming": "₹1,80,000", "total_outgoing": "₹1,79,800", "unique_senders": 1, "unique_receivers": 1, "avg_tx_amount": "₹90,000", "notes": "Shares device with ACCT-401"},
    "ACCT-403": {"account_id": "ACCT-403", "name": "Unknown Holder C", "kyc_status": "Flagged", "account_type": "Savings", "city": "Kolkata", "risk_tier": "High Risk", "open_since": "21 Apr 2025", "device_count": 1, "total_transactions": "4", "total_incoming": "₹3,60,000", "total_outgoing": "₹3,59,000", "unique_senders": 2, "unique_receivers": 2, "avg_tx_amount": "₹90,000", "notes": "Previously flagged in another alert"},
    "ACCT-301": {"account_id": "ACCT-301", "name": "TechServ Solutions Pvt Ltd", "kyc_status": "Verified", "account_type": "Corporate", "city": "Hyderabad", "risk_tier": "Low Risk", "open_since": "Mar 2014", "device_count": 0, "total_transactions": "5,200", "total_incoming": "₹12.4 Cr", "total_outgoing": "₹0", "unique_senders": 340, "unique_receivers": 0, "avg_tx_amount": "₹23,846", "notes": "Verified vendor — regular payment schedule confirmed"},
}

# 2-hop neighbors for graph network
TWO_HOP_NEIGHBORS = {
    "ACCT-801": ["HOP2-8011", "HOP2-8012"],
    "ACCT-802": ["HOP2-8021", "HOP2-8022"],
    "ACCT-803": ["HOP2-8031"],
    "ACCT-804": ["HOP2-8041", "HOP2-8042"],
    "ACCT-805": ["HOP2-8051"],
    "ACCT-901": ["HOP2-9011"],
    "ACCT-902": ["HOP2-9021", "HOP2-9022"],
    "ACCT-401": ["HOP2-4011"],
    "ACCT-402": ["HOP2-4021"],
    "ACCT-403": ["HOP2-4031", "HOP2-4032"],
}

HOP2_PROFILES = {
    "HOP2-8011": {"name": "Ankit Verma", "kyc_status": "Unverified", "risk_tier": "High Risk", "city": "Delhi"},
    "HOP2-8012": {"name": "Priti Saxena", "kyc_status": "Minimal KYC", "risk_tier": "High Risk", "city": "Noida"},
    "HOP2-8021": {"name": "Rohit Kapoor", "kyc_status": "Minimal KYC", "risk_tier": "High Risk", "city": "Gurgaon"},
    "HOP2-8022": {"name": "Sonal Mehta", "kyc_status": "Unverified", "risk_tier": "High Risk", "city": "Delhi"},
    "HOP2-8031": {"name": "Tarun Mishra", "kyc_status": "Minimal KYC", "risk_tier": "High Risk", "city": "Gurgaon"},
    "HOP2-8041": {"name": "Ritu Agarwal", "kyc_status": "Minimal KYC", "risk_tier": "High Risk", "city": "Faridabad"},
    "HOP2-8042": {"name": "Deepak Sood", "kyc_status": "Unverified", "risk_tier": "High Risk", "city": "Delhi"},
    "HOP2-8051": {"name": "Nisha Rao", "kyc_status": "Minimal KYC", "risk_tier": "High Risk", "city": "Delhi"},
    "HOP2-9011": {"name": "Offshore Entity X", "kyc_status": "Flagged", "risk_tier": "High Risk", "city": "Unknown"},
    "HOP2-9021": {"name": "Shell Co. Alpha", "kyc_status": "Flagged", "risk_tier": "High Risk", "city": "Unknown"},
    "HOP2-9022": {"name": "Crypto Exchange Y", "kyc_status": "Unverified", "risk_tier": "High Risk", "city": "Unknown"},
    "HOP2-4011": {"name": "Cash Withdrawal ATM", "kyc_status": "N/A", "risk_tier": "High Risk", "city": "Kolkata"},
    "HOP2-4021": {"name": "Online Gaming Portal", "kyc_status": "Unverified", "risk_tier": "High Risk", "city": "Kolkata"},
    "HOP2-4031": {"name": "Wire Transfer Int'l", "kyc_status": "Flagged", "risk_tier": "High Risk", "city": "Unknown"},
    "HOP2-4032": {"name": "P2P Lending App", "kyc_status": "Unverified", "risk_tier": "Medium Risk", "city": "Kolkata"},
}


# ─── Helper Functions ────────────────────────────────────────────────────────

def get_transactions_df():
    """Returns a pandas DataFrame of transactions."""
    return pd.DataFrame(TRANSACTIONS)

def get_transaction_by_id(tx_id):
    """Retrieve full details for a specific transaction ID."""
    for tx in TRANSACTIONS:
        if tx["tx_id"] == tx_id:
            return tx
    return TRANSACTIONS[0]

def get_customer_profile(account_id):
    """Retrieve sender customer profile metrics and behavior summary."""
    if account_id in CUSTOMER_PROFILES:
        return CUSTOMER_PROFILES[account_id]
    return {
        "account_id": account_id,
        "name": "Unknown",
        "kyc_status": "Unknown",
        "account_type": "Savings",
        "city": "—",
        "risk_tier": "Unknown",
        "open_since": "—",
        "last_login": "—",
        "device_count": 0,
        "total_transactions": "500",
        "total_incoming": "₹50.0 L",
        "total_outgoing": "₹48.0 L",
        "unique_senders": 15,
        "unique_receivers": 20,
        "avg_tx_amount": "₹30,000",
        "behavior_summary": [
            {"metric": "Transactions", "historical": "4", "current": "6", "change": "↑ 50%", "change_type": "medium"},
            {"metric": "Unique Receivers", "historical": "2", "current": "3", "change": "↑ 50%", "change_type": "medium"},
            {"metric": "New Receiver Ratio", "historical": "10%", "current": "30%", "change": "↑ 20pp", "change_type": "medium"},
            {"metric": "Avg. Amount (₹)", "historical": "30,000", "current": "45,000", "change": "↑ 50%", "change_type": "medium"},
        ]
    }

def get_receiver_profile(account_id):
    """Retrieve receiver account profile (for right panel on table row click)."""
    if account_id in RECEIVER_PROFILES:
        return RECEIVER_PROFILES[account_id]
    return {
        "account_id": account_id,
        "name": "Unknown",
        "kyc_status": "Unknown",
        "account_type": "Savings",
        "city": "—",
        "risk_tier": "Unknown",
        "open_since": "—",
        "device_count": 0,
        "total_transactions": "—",
        "total_incoming": "—",
        "total_outgoing": "—",
        "unique_senders": 0,
        "unique_receivers": 0,
        "avg_tx_amount": "—",
        "notes": "No profile data available"
    }

def get_fan_out_rows(tx_id):
    """Returns sub-transaction rows for the fan-out table."""
    tx = get_transaction_by_id(tx_id)
    return tx.get("fan_out_sub_transactions", [])

def get_flagged_transactions():
    """Returns all High and Medium risk transactions (for left panel)."""
    return [tx for tx in TRANSACTIONS if tx["risk"] in ("High", "Medium")]

def get_all_flagged_senders():
    """Returns unique flagged sender accounts with their highest risk transaction."""
    seen = {}
    for tx in TRANSACTIONS:
        acc = tx["account"]
        if acc not in seen or tx["risk_score"] > seen[acc]["risk_score"]:
            seen[acc] = tx
    return list(seen.values())

def get_trend_data():
    """
    Returns historical daily transaction trend data.
    Dates: 29 Apr to 05 May
    Median line: 31K (31,000)
    """
    dates = ["29 Apr", "30 Apr", "1 May", "2 May", "3 May", "4 May", "5 May"]
    amounts = [18000, 42000, 25000, 35000, 15000, 22000, 168000]
    median_val = 31000
    return pd.DataFrame({
        "Date": dates,
        "Amount": amounts,
        "Median": median_val
    })

def create_network_graph(tx_id, include_2hop=True):
    """
    Generates a NetworkX directed graph for the transaction.
    When include_2hop=True, adds 2nd-hop neighbors.
    """
    tx = get_transaction_by_id(tx_id)
    G = nx.DiGraph()

    source_acc = tx["account"]
    pattern = tx["pattern"]

    G.add_node(
        source_acc,
        node_type="source",
        label=f"Source\n{source_acc}",
        color="#ef4444" if tx["risk"] == "High" else "#2563eb",
        hop=0
    )

    if pattern == "Fan-Out":
        for idx, target in enumerate(tx["to_accounts"]):
            G.add_node(
                target,
                node_type="target",
                label=f"Receiver\n{target}",
                color="#f59e0b",
                hop=1
            )
            G.add_edge(source_acc, target, amount=f"₹{tx['amount'] // len(tx['to_accounts']):,}", hop=1)

            if include_2hop and target in TWO_HOP_NEIGHBORS:
                for hop2 in TWO_HOP_NEIGHBORS[target]:
                    G.add_node(
                        hop2,
                        node_type="hop2",
                        label=f"2-Hop\n{hop2}",
                        color="#94a3b8",
                        hop=2
                    )
                    G.add_edge(target, hop2, amount="Transfer", hop=2)

    elif pattern == "Circular Money Flow":
        accs = tx["to_accounts"]
        for i in range(len(accs)):
            u = accs[i]
            v = accs[(i + 1) % len(accs)]
            G.add_node(u, node_type="ring", label=f"Ring\n{u}", color="#dc2626", hop=1)
            G.add_edge(u, v, amount=f"₹{tx['amount']:,}", hop=1)

        if include_2hop:
            for acc in tx["to_accounts"]:
                if acc in TWO_HOP_NEIGHBORS:
                    for hop2 in TWO_HOP_NEIGHBORS[acc]:
                        G.add_node(hop2, node_type="hop2", label=f"2-Hop\n{hop2}", color="#94a3b8", hop=2)
                        G.add_edge(acc, hop2, amount="Transfer", hop=2)

    else:
        for idx, target in enumerate(tx["to_accounts"]):
            G.add_node(target, node_type="target", label=f"Receiver\n{target}", color="#64748b", hop=1)
            G.add_edge(source_acc, target, amount=f"₹{tx['amount']:,}", hop=1)

            if include_2hop and target in TWO_HOP_NEIGHBORS:
                for hop2 in TWO_HOP_NEIGHBORS[target]:
                    G.add_node(hop2, node_type="hop2", label=f"2-Hop\n{hop2}", color="#94a3b8", hop=2)
                    G.add_edge(target, hop2, amount="Transfer", hop=2)

    return G
