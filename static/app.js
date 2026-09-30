// ScamShield Frontend Application Logic
// Preloaded examples representing multiple languages and scripts
const EXAMPLES = [
    {
        label: "English Scam",
        text: "Congratulations! You won a cash reward of ₹15,499 from PhonePe. Claim your prize instantly at http://bit.ly/pay-upi-ref",
        lang: "English"
    },
    {
        label: "Hinglish Scam",
        text: "Jaldi karein! Apna HDFC electricity bill 10 mins ke andar pay karein nahi to aaj raat light cut jayegi. Pay here: http://electricity-bill-pay.com",
        lang: "Hinglish"
    },
    {
        label: "Tanglish Scam",
        text: "Kadaisi warning! 30 mins kulla unga HDFC credit card block aayidum. Secure panna udanae click: https://kyc-update-verification.com/login",
        lang: "Tanglish"
    },
    {
        label: "Hindi Scam",
        text: "बधाई हो! आपने PhonePe की तरफ से ₹15,499 का कैशबैक जीता है। अभी अपने बैंक में ट्रांसफर करें: http://bit.ly/pay-upi-ref",
        lang: "Hindi"
    },
    {
        label: "Tamil Scam",
        text: "உடனே செலுத்துங்கள்! 10 minutesக்குள் உங்கள் HDFC மின்சார கட்டணத்தை செலுத்தவில்லை என்றால் மின்சாரம் துண்டிக்கப்படும். http://electricity-bill-pay.com",
        lang: "Tamil"
    },
    {
        label: "Tenglish Scam",
        text: "Mee HDFC ATM card block chesaru safety reasons valla. Unblock cheyyataniki click cheyyandi: https://kyc-update-verification.com/login",
        lang: "Tenglish"
    },
    {
        label: "Bengali Scam",
        text: "জরুরী! 10 minutes এর মধ্যে আপনার PNB বিদ্যুৎ বিল পরিশোধ করুন না হলে বিদ্যুৎ সংযোগ বিচ্ছিন্ন করা হবে। লিঙ্ক: http://electricity-bill-pay.com",
        lang: "Bengali"
    },
    {
        label: "Legit Bank SMS",
        text: "Dear Customer, your SBI account XXXX4921 has been credited with ₹5,000 via UPI. Ref: 61092819028. - State Bank of India",
        lang: "Legit"
    }
];
let currentMessageId = null;
let tacticChartInstance = null;
let langChartInstance = null;
// Initialize App
document.addEventListener("DOMContentLoaded", () => {
    populateExamples();
    // Pre-initialize stats if tab is active (just in case)
    fetchStatsAndRenderCharts();
});
// Populate Example Pills
function populateExamples() {
    const container = document.getElementById("example-container");
    container.innerHTML = "";
    
    EXAMPLES.forEach(ex => {
        const pill = document.createElement("button");
        pill.className = "example-pill";
        pill.textContent = ex.label;
        pill.onclick = () => {
            document.getElementById("message-input").value = ex.text;
            document.getElementById("message-input").focus();
        };
        container.appendChild(pill);
    });
}
// Clipboard Paste
async function pasteFromClipboard() {
    try {
        const text = await navigator.clipboard.readText();
        if (text) {
            document.getElementById("message-input").value = text;
        }
    } catch (err) {
        console.error("Failed to read clipboard: ", err);
    }
}
// Clear Input
function clearInput() {
    document.getElementById("message-input").value = "";
    const resContainer = document.getElementById("result-container");
    const resDetails = document.getElementById("result-details");
    const emptyState = document.getElementById("result-empty-state");
    
    resContainer.className = "card result-card empty";
    resDetails.classList.add("hide");
    emptyState.classList.remove("hide");
    currentMessageId = null;
}
// Submit Analysis
async function submitAnalysis() {
    const text = document.getElementById("message-input").value.trim();
    if (!text) return;
    
    const submitBtn = document.getElementById("analyze-submit-btn");
    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");
    
    // Loading State
    submitBtn.disabled = true;
    btnText.style.opacity = "0.5";
    btnSpinner.style.display = "block";
    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        
        const data = await response.json();
        renderResults(text, data);
    } catch (err) {
        console.error("Analysis failed:", err);
        alert("Failed to analyze message. Make sure the backend server is running.");
    } finally {
        submitBtn.disabled = false;
        btnText.style.opacity = "1";
        btnSpinner.style.display = "none";
    }
}
// Render Results
function renderResults(originalText, data) {
    currentMessageId = data.message_id;
    
    const resContainer = document.getElementById("result-container");
    const emptyState = document.getElementById("result-empty-state");
    const resDetails = document.getElementById("result-details");
    
    emptyState.classList.add("hide");
    resDetails.classList.remove("hide");
    
    // Set general card style based on prediction
    const isScam = data.label === "scam";
    resContainer.className = `card result-card ${isScam ? 'border-scam' : 'border-safe'}`;
    
    // Set Badge
    const riskBadge = document.getElementById("risk-badge");
    riskBadge.textContent = isScam ? "SCAM FLAGGED" : "SAFE / TRUSTED";
    riskBadge.className = `badge ${isScam ? 'scam' : 'safe'}`;
    
    // Set Language & Script Badge
    const langBadge = document.getElementById("meta-lang-badge");
    langBadge.textContent = `Language: ${data.language_detected} (Script: ${data.script_type})`;
    
    // Set Message ID
    document.getElementById("msg-id-display").textContent = `ID: ${data.message_id.substring(0, 8)}`;
    
    // Set Meter Bar
    const meterPercentage = document.getElementById("scam-percentage");
    const meterBar = document.getElementById("scam-meter-bar");
    const pct = Math.round(data.scam_probability * 100);
    
    meterPercentage.textContent = `${pct}%`;
    meterBar.style.width = `${pct}%`;
    
    // Meter color shifting
    if (pct < 30) {
        meterBar.style.backgroundColor = "var(--color-safe)";
    } else if (pct < 70) {
        meterBar.style.backgroundColor = "var(--color-warning)";
    } else {
        meterBar.style.backgroundColor = "var(--color-danger)";
    }
    // Set Tactics
    const tacticsSection = document.getElementById("detected-tactics-section");
    const tacticGrid = document.getElementById("tactic-grid");
    tacticGrid.innerHTML = "";
    
    if (isScam && data.tactics && data.tactics.length > 0) {
        tacticsSection.classList.remove("hide");
        
        data.tactics.forEach(t => {
            const pill = document.createElement("div");
            pill.className = "tactic-pill";
            
            const tacticTitle = t.tactic.replace("_", " ").toUpperCase();
            
            pill.innerHTML = `
                <div class="tactic-name-area">
                    <svg class="tactic-bullet-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                    <div>
                        <div class="tactic-name">${escapeHtml(tacticTitle)}</div>
                        <div class="tactic-trigger">Evidence: "${escapeHtml(t.evidence)}"</div>
                    </div>
                </div>
                <div class="tactic-confidence">Conf: ${Math.round(t.confidence * 100)}%</div>
            `;
            tacticGrid.appendChild(pill);
        });
    } else {
        tacticsSection.classList.add("hide");
    }
    
    // Highlight Evidence in the Original Text Area and Set Explanation
    // NOTE: the message text and the evidence spans come from the sender, so they are
    // HTML-escaped before any markup is added. Escaping first means the <mark> tags
    // below are the only markup in the output and nothing a sender wrote can execute.
    let highlightedText = escapeHtml(originalText);
    if (isScam && data.tactics) {
        // Find evidence phrases and wrap them with <mark>
        // Sort evidence by length descending to avoid nested replacement issues
        const uniqueEvidence = [...new Set(data.tactics.map(t => t.evidence))]
            .filter(e => e && e !== "suspicious phrasing")
            .sort((a, b) => b.length - a.length);
        uniqueEvidence.forEach(ev => {
            try {
                // Case-insensitive regex highlighting match against the escaped text
                const escaped = escapeHtml(ev);
                const regex = new RegExp(`(${escapeRegExp(escaped)})`, 'gi');
                highlightedText = highlightedText.replace(regex, "<mark>$1</mark>");
            } catch (e) {
                console.error("Regex highlight error: ", e);
            }
        });
    }
    // Set Explanation Box Content
    const expBox = document.getElementById("explanation-box");
    // Show highlighted message text followed by the generated plain-language text
    expBox.innerHTML = `
        <p class="highlighted-msg-preview"><strong>Input Preview:</strong> <em>${highlightedText}</em></p>
        <div class="explanation-paragraphs">
            ${escapeHtml(data.explanation).replace(/\n/g, "<br>")}
        </div>
    `;
    // Reset feedback button state
    resetFeedbackButtons();
}
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
function escapeHtml(string) {
    return String(string ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}
// Send Feedback
async function sendFeedback(correction) {
    if (!currentMessageId) return;
    
    const legitBtn = document.querySelector(".feedback-legit");
    const scamBtn = document.querySelector(".feedback-scam");
    
    legitBtn.disabled = true;
    scamBtn.disabled = true;
    
    try {
        const response = await fetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message_id: currentMessageId,
                user_correction: correction
            })
        });
        
        if (response.ok) {
            showToast(`Feedback logged: marked as ${correction}`);
            // Disable clicked option
            if (correction === "legit") {
                legitBtn.classList.add("active-feedback");
                legitBtn.textContent = "Safe Logged";
            } else {
                scamBtn.classList.add("active-feedback");
                scamBtn.textContent = "Scam Logged";
            }
        }
    } catch (err) {
        console.error("Feedback failed:", err);
    }
}
function resetFeedbackButtons() {
    const legitBtn = document.querySelector(".feedback-legit");
    const scamBtn = document.querySelector(".feedback-scam");
    
    legitBtn.disabled = false;
    scamBtn.disabled = false;
    legitBtn.classList.remove("active-feedback");
    scamBtn.classList.remove("active-feedback");
    
    legitBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="f-icon"><path d="M22 11.08V12a10 10 0 11-5.93-9.14M22 4L12 14.01l-3-3"/></svg>
        Mark as Safe
    `;
    scamBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="f-icon"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01"/></svg>
        Mark as Scam
    `;
}
function showToast(message) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.classList.add("show");
    
    setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}
// Tab Switching
function switchTab(tabName) {
    // Nav buttons active toggle
    const buttons = document.querySelectorAll(".nav-btn");
    buttons.forEach(btn => btn.classList.remove("active"));
    
    const activeBtn = document.getElementById(`tab-${tabName}-btn`);
    if (activeBtn) activeBtn.classList.add("active");
    
    // Panel display toggle
    const panels = document.querySelectorAll(".tab-panel");
    panels.forEach(panel => panel.classList.remove("active"));
    
    const activePanel = document.getElementById(`tab-${tabName}`);
    if (activePanel) activePanel.classList.add("active");
    
    // Refresh stats if stats tab is opened
    if (tabName === "stats") {
        fetchStatsAndRenderCharts();
    }
}
// Fetch stats and render charts
async function fetchStatsAndRenderCharts() {
    try {
        const response = await fetch("/api/stats");
        if (!response.ok) return;
        
        const stats = await response.json();
        
        // Update overview cards
        document.getElementById("stat-total").textContent = stats.total_analyzed || 0;
        document.getElementById("stat-scams").textContent = (stats.label_distribution && stats.label_distribution.scam) || 0;
        document.getElementById("stat-legit").textContent = (stats.label_distribution && stats.label_distribution.legit) || 0;
        document.getElementById("stat-feedback").textContent = stats.feedback_received || 0;
        
        // Update Web vs. SMS breakdown
        if (stats.source_distribution) {
            document.getElementById("stat-web").textContent = stats.source_distribution.web || 0;
            document.getElementById("stat-sms").textContent = stats.source_distribution.sms || 0;
        }
        
        // Render Tactic Chart
        renderTacticChart(stats.tactic_distribution || {});
        
        // Render Language Chart
        renderLanguageChart(stats.language_distribution || {});
        
        // Render Script Breakdown Bar
        renderScriptBreakdown(stats.script_distribution || {});
    } catch (err) {
        console.error("Failed to load dashboard statistics:", err);
    }
}
// Render Tactic Frequency Chart (Horizontal Bar Chart)
function renderTacticChart(data) {
    const ctx = document.getElementById("tacticChart").getContext("2d");
    
    const labels = Object.keys(data);
    const counts = Object.values(data);
    
    // Set fallback sample if no entries exist
    const displayLabels = labels.length > 0 ? labels : ["Urgency", "Authority Impersonation", "False Reward", "Loss Aversion", "Phishing", "Suspicious Link"];
    const displayCounts = counts.length > 0 ? counts : [0, 0, 0, 0, 0, 0];
    
    if (tacticChartInstance) {
        tacticChartInstance.destroy();
    }
    
    tacticChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: displayLabels,
            datasets: [{
                label: 'Occurrences',
                data: displayCounts,
                backgroundColor: 'rgba(239, 68, 68, 0.45)',
                borderColor: '#ef4444',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', precision: 0 }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}
// Render Language Distribution Chart (Doughnut Chart)
function renderLanguageChart(data) {
    const ctx = document.getElementById("langChart").getContext("2d");
    
    const labels = Object.keys(data);
    const counts = Object.values(data);
    
    const displayLabels = labels.length > 0 ? labels : ["English", "Hindi", "Tamil", "Telugu", "Bengali"];
    const displayCounts = counts.length > 0 ? counts : [1, 0, 0, 0, 0]; // default showing English to keep shape
    
    if (langChartInstance) {
        langChartInstance.destroy();
    }
    
    const colors = [
        '#00f2fe',  // Cyan
        '#3b82f6',  // Blue
        '#10b981',  // Green
        '#f59e0b',  // Amber
        '#ec4899'   // Pink
    ];
    
    langChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: displayLabels,
            datasets: [{
                data: displayCounts,
                backgroundColor: colors.slice(0, displayLabels.length),
                borderColor: '#070b13',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#9ca3af',
                        font: { family: 'Inter', size: 11 }
                    }
                }
            },
            cutout: '65%'
        }
    });
}
// Render Script Segment bar
function renderScriptBreakdown(data) {
    const bar = document.getElementById("script-breakdown-bar");
    const labelContainer = document.getElementById("script-labels");
    
    bar.innerHTML = "";
    labelContainer.innerHTML = "";
    
    const total = Object.values(data).reduce((a, b) => a + b, 0);
    
    const categories = [
        { key: "Native", class: "native", color: "#00f2fe" },
        { key: "Romanized", class: "romanized", color: "#3b82f6" },
        { key: "Latin", class: "latin", color: "#6b7280" }
    ];
    
    categories.forEach(cat => {
        const count = data[cat.key] || 0;
        const pct = total > 0 ? (count / total) * 100 : 0;
        
        // Add bar segment if pct > 0
        if (pct > 0 || (total === 0 && cat.key === "Latin")) {
            const displayPct = total === 0 ? 100 : pct;
            const segment = document.createElement("div");
            segment.className = `script-bar-segment ${cat.class}`;
            segment.style.width = `${displayPct}%`;
            segment.title = `${cat.key}: ${total === 0 ? 0 : count} messages (${Math.round(displayPct)}%)`;
            bar.appendChild(segment);
        }
        
        // Add Label Item
        const labelItem = document.createElement("div");
        labelItem.className = "script-label-item";
        labelItem.innerHTML = `
            <div class="script-color-box ${cat.class}"></div>
            <span>${cat.key}: ${total === 0 ? 0 : count} (${total === 0 ? 0 : Math.round(pct)}%)</span>
        `;
        labelContainer.appendChild(labelItem);
    });
}
