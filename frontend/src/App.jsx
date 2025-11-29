import { useState } from 'react'
import axios from 'axios'
import { 
  UploadCloud, 
  FileText, 
  ShieldCheck, 
  Lock, 
  Zap, 
  Search, 
  X, 
  Loader2,
  Info,
  Sparkles,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react'
import './App.css'

function App() {
  const [inputText, setInputText] = useState("")
  const [selectedFile, setSelectedFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const API_BASE_URL = "https://contract-sentinel-api.onrender.com" 

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setSelectedFile(e.target.files[0])
      setInputText("") 
      setResult(null)
      setError("")
    }
  }

  const handleTextChange = (e) => {
    setInputText(e.target.value)
    setSelectedFile(null)
    setResult(null)
    setError("")
  }

  const insertExample = () => {
    const example = "Consultant acknowledges that payment is contingent upon the Company's receipt of funds from the End Client. If the End Client fails to pay, Company shall have no obligation to remit fees to Consultant.";
    setInputText(example);
    setSelectedFile(null);
    setResult(null);
  }

  const analyzeContract = async () => {
    if (!inputText && !selectedFile) {
      setError("Please upload a PDF or paste a contract clause.")
      return
    }

    setLoading(true)
    setError("")
    setResult(null)

    try {
      let response;
      if (selectedFile) {
        const formData = new FormData();
        formData.append("file", selectedFile);
        response = await axios.post(`${API_BASE_URL}/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
      } else {
        response = await axios.post(`${API_BASE_URL}/analyze`, { text: inputText });
      }
      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError("Service unavailable. Please ensure the backend is running.")
    } finally {
      setLoading(false)
    }
  }

  // --- PREMIUM THEME SYSTEM ---
  const getTheme = (level) => {
    switch (level) {
      case "Critical": 
        return { 
          primary: "#BE123C", 
          bg: "#FFF1F2", // Subtle tint for card bg
          border: "#FECACA", 
          icon: <AlertTriangle size={22} color="#BE123C" />,
          // Point 3: Gradient for progress bar
          barGradient: "linear-gradient(90deg, #FB7185 0%, #E11D48 100%)"
        };
      case "High": 
        return { 
          primary: "#C2410C", 
          bg: "#FFF7ED",
          border: "#FED7AA", 
          icon: <AlertTriangle size={22} color="#C2410C" />,
          barGradient: "linear-gradient(90deg, #FB923C 0%, #EA580C 100%)"
        };
      case "Medium": 
        return { 
          primary: "#B45309", 
          bg: "#FEFCE8",
          border: "#FDE68A", 
          icon: <Info size={22} color="#B45309" />,
          barGradient: "linear-gradient(90deg, #FACC15 0%, #CA8A04 100%)"
        };
      case "Low": 
        return { 
          primary: "#1D4ED8", 
          bg: "#EFF6FF",
          border: "#BFDBFE", 
          icon: <Info size={22} color="#1D4ED8" />,
          barGradient: "linear-gradient(90deg, #60A5FA 0%, #2563EB 100%)"
        };
      case "Safe": 
        return { 
          primary: "#047857", 
          bg: "#F0FDF4",
          border: "#BBF7D0", 
          icon: <CheckCircle2 size={22} color="#047857" />,
          barGradient: "linear-gradient(90deg, #34D399 0%, #059669 100%)"
        };
      default: 
        return { 
          primary: "#374151", 
          bg: "#F9FAFB", 
          border: "#E5E7EB", 
          icon: <Info size={22} />,
          barGradient: "#9CA3AF"
        };
    }
  }

  const theme = result ? getTheme(result.risk_level) : null;

  return (
    <div className="app-container">
      
      {/* HEADER */}
      <header>
        {/* Point 11: Premium Badge */}
        <div className="beta-badge">
          <div className="dot"></div>
          <span>Prototype • Not Legal Advice</span>
        </div>
        
        <div>
          <h1>
            <ShieldCheck className="logo-icon" size={36} strokeWidth={2.5} color="#1E3DF6" />
            ContractSentinel
          </h1>
          <p className="tagline">
            AI-powered legal intelligence for freelancers. 
          </p>
        </div>
      </header>

      {/* INPUT CARD */}
      <main className="main-card">
        
        {/* Upload */}
        <label className="upload-box">
          <input type="file" accept=".pdf" onChange={handleFileChange} hidden />
          <UploadCloud size={40} strokeWidth={1.5} color="#2563EB" />
          <div style={{textAlign: 'center'}}>
            <div className="upload-title">Click to upload PDF</div>
            <div className="upload-desc">Up to 5MB • Secure Processing</div>
          </div>
        </label>

        {selectedFile && (
          <div className="file-row">
            <div className="file-name">
              <FileText size={16} color="#1E3DF6" />
              <span>{selectedFile.name}</span>
            </div>
            <button onClick={() => setSelectedFile(null)} className="remove-btn">
              <X size={18} />
            </button>
          </div>
        )}

        {/* Point 12: Fainter Divider */}
        <div className="divider">
          <div className="line"></div>
          <span className="divider-label">OR PASTE TEXT</span>
          <div className="line"></div>
        </div>

        {/* Text Area */}
        <div className="text-wrapper">
          <textarea
            className="text-area"
            placeholder="Paste contract clause here..."
            value={inputText}
            onChange={handleTextChange}
            disabled={!!selectedFile}
          />
          {!inputText && !selectedFile && (
            <div className="example-trigger">
               <Sparkles size={14} color="#1E3DF6" />
               <span className="trigger-link" onClick={insertExample}>
                 Try Example: "Ghost Pay" Clause
               </span>
            </div>
          )}
        </div>

        {/* Point 7: Larger CTA */}
        <button 
          className="primary-btn" 
          onClick={analyzeContract} 
          disabled={loading}
        >
          {loading ? (
            <> <Loader2 className="spin" size={18} /> Analyzing... </>
          ) : (
            <> <Search size={18} /> Analyze Risk </>
          )}
        </button>

        {error && <div className="error-toast">{error}</div>}

        {/* Trust Footer */}
        <div className="trust-row">
          <div className="trust-item">
            <Lock size={14} /> TLS Encrypted
          </div>
          <div className="trust-item">
            <Zap size={14} /> Instant Analysis
          </div>
          <div className="trust-item">
             <ShieldCheck size={14} /> Privacy First
          </div>
        </div>
      </main>

      {/* RESULTS SECTION */}
      {result && (
        <div className="results-wrapper">
          {/* Point 9: Refined Label */}
          <span className="section-label">Analysis Result</span>
          
          <div 
            className="risk-card"
            style={{ 
              backgroundColor: theme.bg,
              borderLeftColor: theme.primary
            }}
          >
            <div className="card-header">
              {/* Point 4: Increased Spacing */}
              <div className="card-title-group">
                {theme.icon}
                <h2 className="card-title">{result.category}</h2>
              </div>
              
              <span 
                className="card-badge"
                style={{ 
                  color: theme.primary, 
                  borderColor: theme.border,
                  backgroundColor: "#FFFFFF"
                }}
              >
                {result.risk_level} RISK
              </span>
            </div>
            
            <div className="meter-container">
              {/* Point 6: Smaller Label */}
              <div className="meter-label">
                <span>AI Confidence</span>
                <span>{(result.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="meter-track">
                <div 
                  className="meter-fill" 
                  style={{ 
                    width: `${result.confidence * 100}%`,
                    background: theme.barGradient /* Point 3: Gradient Fill */
                  }}
                ></div>
              </div>
            </div>

            {/* Point 5: Darker Gray Body */}
            <p className="card-text">{result.description}</p>
          </div>
        </div>
      )}

      {/* Point 14: Increased Spacing */}
      <footer>
        © 2024 ContractSentinel • Built for Education
      </footer>
    </div>
  )
}

export default App