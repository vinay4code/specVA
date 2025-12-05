import React, { useState, useEffect } from "react";
import { Upload, Zap, BarChart3, FileText, Check, AlertCircle } from "lucide-react";

// --- UI Components (Simulating the shadcn/ui components used in the original) ---

const Card = ({ className, children }) => (
  <div className={`rounded-xl shadow-sm ${className}`}>
    {children}
  </div>
);

const Button = ({ className, onClick, disabled, children }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${className}`}
  >
    {children}
  </button>
);

// --- Main Application Component ---

export default function SpectroscopyApp() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("spectral");

  // Handler for file drag over
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  // Handler for file drag leave
  const handleDragLeave = () => {
    setIsDragging(false);
  };

  // Handler for file drop
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
        // Allowing loose checking for demo purposes, but prioritizing .fits logic
        // In a real app, you might want to show an error if it's not .fits
        if (droppedFile.name.endsWith(".fits") || droppedFile.name.endsWith(".FITS")) {
            setFile(droppedFile);
            setResults(null); // Reset results on new file
        } else {
            alert("Please upload a valid .fits file");
        }
    }
  };

  // Handler for file input click
  const handleFileInput = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
        if (selectedFile.name.endsWith(".fits") || selectedFile.name.endsWith(".FITS")) {
            setFile(selectedFile);
            setResults(null);
        } else {
            alert("Please upload a valid .fits file");
        }
    }
  };

  // Simulation of the analysis process
  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    
    // Simulate complex calculation/API delay
    setTimeout(() => {
      setResults({
        wavelengthStart: 3800,
        wavelengthEnd: 9200,
        peakWavelength: 6562.8,
        intensity: 2450.5,
        elementalComposition: [
          { element: "Hydrogen", percentage: 73.46 },
          { element: "Helium", percentage: 24.85 },
          { element: "Other", percentage: 1.69 },
        ],
      });
      setLoading(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 font-sans text-slate-200">
      {/* Background Star Effect */}
      <div
        className="fixed inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage:
            "radial-gradient(2px 2px at 20% 30%, white, rgba(0,0,0,0)), radial-gradient(2px 2px at 60% 70%, white, rgba(0,0,0,0)), radial-gradient(1px 1px at 50% 50%, white, rgba(0,0,0,0))",
          backgroundSize: "100% 100%",
        }}
      ></div>

      <div className="max-w-6xl mx-auto relative z-10">
        {/* Header */}
        <div className="mb-8 animate-in fade-in slide-in-from-top-4 duration-700">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-cyan-600 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-cyan-300 to-blue-400 bg-clip-text text-transparent">
              Spectroscopy Analysis
            </h1>
          </div>
          <p className="text-slate-400 text-lg">Advanced astronomical spectral analysis dashboard</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content Column */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Upload Section */}
            <Card className="border border-slate-700 bg-slate-900/50 backdrop-blur-sm hover:border-slate-600 transition-colors duration-300">
              <div className="p-6">
                <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                  <Upload className="w-5 h-5 text-cyan-400" />
                  Upload FITS File
                </h2>

                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`relative group border-2 border-dashed rounded-lg p-10 text-center transition-all duration-300 ease-in-out cursor-pointer ${
                    isDragging
                      ? "border-cyan-400 bg-cyan-400/10 scale-[1.01]"
                      : "border-slate-600 bg-slate-800/30 hover:border-slate-500 hover:bg-slate-800/50"
                  }`}
                >
                  <input 
                    type="file" 
                    accept=".fits" 
                    onChange={handleFileInput} 
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20" 
                    id="file-input" 
                  />
                  
                  <div className="relative z-10 pointer-events-none">
                    <div className={`mb-4 transition-transform duration-300 ${isDragging ? 'scale-110' : 'group-hover:scale-105'}`}>
                      <div className="w-16 h-16 mx-auto rounded-full bg-slate-800 flex items-center justify-center border border-slate-700">
                        <Upload className="w-8 h-8 text-cyan-400 opacity-80" />
                      </div>
                    </div>
                    <p className="text-white font-medium text-lg">
                      {file ? file.name : "Drag and drop your FITS file here"}
                    </p>
                    <p className="text-slate-400 text-sm mt-2">or click to browse local files</p>
                  </div>
                </div>

                {file && (
                  <div className="mt-4 p-3 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-sm flex items-center gap-2 animate-in fade-in zoom-in-95">
                    <Check className="w-4 h-4" />
                    <span className="font-medium">File ready for analysis:</span>
                    <span className="opacity-75">{file.name}</span>
                  </div>
                )}
              </div>
            </Card>

            {/* Analysis Results */}
            {results && (
              <Card className="border border-slate-700 bg-slate-900/50 backdrop-blur-sm animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="p-6">
                  <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-orange-400" />
                    Analysis Results
                  </h2>

                  {/* Custom Tabs */}
                  <div className="w-full">
                    <div className="grid grid-cols-2 p-1 bg-slate-800 rounded-lg border border-slate-700 mb-6">
                      <button
                        onClick={() => setActiveTab("spectral")}
                        className={`py-2 px-4 rounded-md text-sm font-medium transition-all duration-200 ${
                          activeTab === "spectral"
                            ? "bg-slate-700 text-cyan-400 shadow-sm ring-1 ring-cyan-500/20"
                            : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/50"
                        }`}
                      >
                        Spectral Data
                      </button>
                      <button
                        onClick={() => setActiveTab("composition")}
                        className={`py-2 px-4 rounded-md text-sm font-medium transition-all duration-200 ${
                          activeTab === "composition"
                            ? "bg-slate-700 text-orange-400 shadow-sm ring-1 ring-orange-500/20"
                            : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/50"
                        }`}
                      >
                        Composition
                      </button>
                    </div>

                    {/* Tab Content: Spectral */}
                    {activeTab === "spectral" && (
                      <div className="animate-in fade-in slide-in-from-left-2 duration-300">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700 hover:border-slate-600 transition-colors">
                            <p className="text-slate-400 text-sm mb-1">Wavelength Start</p>
                            <p className="text-white font-mono font-semibold text-xl">{results.wavelengthStart} Å</p>
                          </div>
                          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700 hover:border-slate-600 transition-colors">
                            <p className="text-slate-400 text-sm mb-1">Wavelength End</p>
                            <p className="text-white font-mono font-semibold text-xl">{results.wavelengthEnd} Å</p>
                          </div>
                          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700 hover:border-cyan-900/50 transition-colors group">
                            <p className="text-slate-400 text-sm mb-1">Peak Wavelength</p>
                            <p className="text-cyan-400 font-mono font-semibold text-xl group-hover:text-cyan-300">{results.peakWavelength} Å</p>
                          </div>
                          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700 hover:border-orange-900/50 transition-colors group">
                            <p className="text-slate-400 text-sm mb-1">Peak Intensity</p>
                            <p className="text-orange-400 font-mono font-semibold text-xl group-hover:text-orange-300">{results.intensity}</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Tab Content: Composition */}
                    {activeTab === "composition" && (
                      <div className="space-y-4 animate-in fade-in slide-in-from-right-2 duration-300">
                        {results.elementalComposition.map((item, idx) => (
                          <div key={idx} className="space-y-2 group">
                            <div className="flex justify-between items-center">
                              <span className="text-white font-medium flex items-center gap-2">
                                <span className={`w-2 h-2 rounded-full ${idx === 0 ? 'bg-cyan-400' : idx === 1 ? 'bg-blue-500' : 'bg-slate-500'}`}></span>
                                {item.element}
                              </span>
                              <span className="text-cyan-400 font-mono font-semibold">{item.percentage}%</span>
                            </div>
                            <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700/50">
                              <div
                                className="h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 rounded-full transition-all duration-1000 ease-out"
                                style={{ width: `${item.percentage}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Action Button */}
            <Button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className={`w-full h-14 text-lg shadow-lg shadow-cyan-900/20 transition-all duration-300 ${
                !file || loading 
                 ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                 : "bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white transform hover:-translate-y-0.5"
              }`}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing Spectrum...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                    <Zap className="w-5 h-5 fill-current" />
                    Analyze Spectrum
                </span>
              )}
            </Button>

            {/* Info Cards */}
            <Card className="border border-slate-700 bg-slate-900/50 backdrop-blur-sm">
              <div className="p-5">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <FileText className="w-4 h-4 text-cyan-400" />
                    Supported Formats
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed border-l-2 border-slate-700 pl-3">
                  Upload FITS files for spectroscopic analysis. Our advanced algorithms detect elemental signatures and
                  wavelength patterns automatically.
                </p>
              </div>
            </Card>

            <Card className="border border-slate-700 bg-slate-900/50 backdrop-blur-sm">
              <div className="p-5">
                <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-orange-400" />
                    Quick Tips
                </h3>
                <ul className="space-y-3 text-slate-400 text-sm">
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-500 mt-1">•</span> 
                    <span>Ensure FITS file is properly formatted with header data</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-500 mt-1">•</span> 
                    <span>Analysis completes in approx 1.5 seconds</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-500 mt-1">•</span> 
                    <span>Results include full spectral breakdown and chemical composition</span>
                  </li>
                </ul>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
