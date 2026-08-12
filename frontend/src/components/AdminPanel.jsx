import React, { useState, useEffect } from 'react';
import { FileText, Upload, Trash2, CheckCircle2, AlertCircle, Loader2, AlertTriangle } from 'lucide-react';

const API_URL = "http://127.0.0.1:8000";

export default function AdminPanel() {
  const [documents, setDocuments] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState({ type: '', message: '' });

  // Stan Modala Potwierdzenia Usuwania
  const [deleteModal, setDeleteModal] = useState({ show: false, fileName: '' });

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents?t=${Date.now()}`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error("Błąd pobierania dokumentów:", err);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setStatus({ type: '', message: '' });
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setStatus({ type: 'error', message: 'Wybierz najpierw plik PDF!' });
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setLoading(true);
    setStatus({ type: 'info', message: 'Trwa indeksowanie pliku w bazie wektorowej...' });

    try {
      const res = await fetch(`${API_URL}/documents`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        setStatus({ type: 'success', message: 'Dokument zaindeksowany pomyślnie!' });
        setSelectedFile(null);
        
        const fileInput = document.getElementById('fileInput');
        if (fileInput) fileInput.value = '';

        await fetchDocuments();
      } else {
        const err = await res.json();
        const errorMessage = typeof err.detail === 'string' 
          ? err.detail 
          : 'Błąd wgrywania pliku.';

        if (res.status === 400){
          setStatus({ type: 'warning', message: errorMessage})
        } else {  
          setStatus({ type: 'error', message: errorMessage });
        }
      }
    } catch (err) {
      setStatus({ type: 'error', message: 'Brak połączenia z serwerem API.' });
    } finally {
      setLoading(false);
    }
  };

  // Pomocnicza funkcja do bezpiecznego odczytu nazwy pliku
  const getDocName = (doc) => {
    if (typeof doc === 'string') return doc;
    if (typeof doc === 'object' && doc !== null) return doc.filename || doc.name || doc.title || '';
    return '';
  };

  const confirmDelete = (doc) => {
    const fileName = getDocName(doc);
    if (!fileName) {
      setStatus({ type: 'error', message: 'Błąd: Nie można odczytać nazwy pliku.' });
      return;
    }
    setDeleteModal({ show: true, fileName });
  };

  const executeDelete = async () => {
    const fileNameToDelete = deleteModal.fileName;

    if (!fileNameToDelete) {
      setStatus({ type: 'error', message: 'Błąd: Brak wskazanej nazwy pliku.' });
      setDeleteModal({ show: false, fileName: '' });
      return;
    }

    setDeleteModal({ show: false, fileName: '' });

    try {
      const res = await fetch(`${API_URL}/documents/${encodeURIComponent(fileNameToDelete)}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        setStatus({ type: 'success', message: `Usunięto plik: ${fileNameToDelete}` });
        await fetchDocuments();
      } else {
        const err = await res.json();
        // Zabezpieczenie przed obiektem w err.detail
        const errorMessage = typeof err.detail === 'string' 
          ? err.detail 
          : JSON.stringify(err.detail) || 'Nie udało się usunąć pliku.';
        setStatus({ type: 'error', message: errorMessage });
      }
    } catch (err) {
      setStatus({ type: 'error', message: 'Błąd połączenia z serwerem podczas usuwania.' });
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 bg-white rounded-xl shadow-xl my-8 relative">
      <h1 className="text-3xl font-bold text-slate-800 mb-6 flex items-center gap-2 ">
        <FileText className="text-indigo-400 w-8 h-8" /> Panel Bazy Wiedzy
      </h1>

      {/* SEKCJA UPLOADU */}
      <div className="border-3 border-dashed border-indigo-400 rounded-xl p-10 text-center bg-indigo-50/50 mb-15">
        <input 
          type="file" 
          id="fileInput" 
          accept=".pdf" 
          className="hidden" 
          onChange={handleFileChange} 
        />
        
        <label 
          htmlFor="fileInput" 
          className="cursor-pointer inline-flex flex-col items-center justify-center"
        >
          <Upload className="w-12 h-12 text-indigo-400 mb-2" />
          <span className="text-slate-1000 font-medium">
            {selectedFile ? selectedFile.name : "Kliknij, aby wybrać plik PDF"}
          </span>
          <span className="text-xs text-slate-500 mt-1">Obsługiwany format: PDF</span>
        </label>

        {selectedFile && (
          <div className="mt-4">
            <button
              onClick={handleUpload}
              disabled={loading}
              className="px-6 py-2 bg-indigo-500 hover:bg-indigo-800 text-white font-medium rounded-full shadow transition-all disabled:opacity-100 flex items-center gap-2 mx-auto cursor-pointer"
            >
              {loading ? <Loader2 className="animate-spin w-6 h-6" /> : "Zaindeksuj plik"}
            </button>
          </div>
        )}

        {/* KOMUNIKATY STATUSU */}
        {status.message && (
          <div className={`mt-4 p-3 rounded-lg text-sm flex items-center justify-center gap-2 ${
            status.type === 'success' ? 'bg-green-100 text-black-700' :
            status.type === 'warning' ? 'bg-amber-50 text-black-700 border border-amber-500' :
            status.type === 'error' ? 'bg-red-100 text-black-700' : 'bg-blue-100 text-black-700'
          }`}>
            {status.type === 'success' && <CheckCircle2 className="w-5 h-5" />}
            {status.type === 'warning' && <AlertTriangle className='w-4 h-4'/>}
            {status.type === 'error' && <AlertCircle className="w-5 h-5" />}
            {status.message}
          </div>
        )}
      </div>

      {/* LISTA DOKUMENTÓW */}
      <div>
        <h2 className="text-xl font-bold text-slate-700 mb-4">Dodane dokumenty</h2>
        {documents.length === 0 ? (
          <p className="text-slate-400 text-sm italic">Brak dokumentów w bazie wiedzy.</p>
        ) : (
          <div className="space-y-4">
            {documents.map((doc, idx) => {
              const fileName = getDocName(doc);
              return (
                <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 border border-slate-200 rounded-lg">
                  <span className="font-medium text-slate-800 flex items-center gap-3">
                    <FileText className="w-5 h-6 text-indigo-400" /> {fileName || "Nieznany dokument"}
                  </span>
                  <button
                    onClick={() => confirmDelete(doc)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors cursor-pointer"
                    title="Usuń dokument"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* MODAL POTWIERDZENIA USUNIĘCIA */}
      {deleteModal.show && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-lg border border-slate-200">
            <div className="flex items-center gap-3 text-red-600 mb-3">
              <AlertTriangle className="w-7 h-7" />
              <h3 className="text-lg font-bold">Potwierdź usunięcie</h3>
            </div>
            <p className="text-slate-600 text-sm mb-6">
              Czy na pewno chcesz usunąć plik <strong className="text-slate-800">{deleteModal.fileName}</strong> z bazy wiedzy? Tej operacji nie będzie można cofnąć.
            </p>
            <div className="flex justify-end gap-4">
              <button
                onClick={() => setDeleteModal({ show: false, fileName: '' })}
                className="px-4 py-2 bg-slate-100 hover:bg-slate-300 text-slate-800 font-medium rounded-lg text-sm transition-colors cursor-pointer"
              >
                Anuluj
              </button>
              <button
                onClick={executeDelete}
                className="px-4 py-2 bg-red-600 hover:bg-red-800 text-white font-medium rounded-lg text-sm transition-colors shadow cursor-pointer"
              >
                Usuń z bazy
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}