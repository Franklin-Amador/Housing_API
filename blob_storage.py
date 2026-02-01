"""
Módulo para manejar descargas desde Vercel Blob Storage
"""

import os
import io
import requests
from typing import BinaryIO, Optional
from urllib.parse import urljoin

class VercelBlobStorage:
    """Cliente para descargar archivos desde Vercel Blob Storage"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Inicializa el cliente de Vercel Blob
        
        Args:
            token: Token BLOB_READ_WRITE_TOKEN (opcional, no necesario para URLs públicas)
        """
        self.token = token or os.getenv("BLOB_READ_WRITE_TOKEN", "")
        
    def download_file(self, blob_url: str) -> bytes:
        """
        Descarga un archivo desde Vercel Blob usando URL pública
        
        Args:
            blob_url: URL completa del archivo en el blob 
                     (ej: 'https://xxx.public.blob.vercel-storage.com/model.pkl')
            
        Returns:
            Contenido del archivo en bytes
            
        Raises:
            Exception: Si hay error descargando el archivo
        """
        try:
            # Para URLs públicas no necesitamos autenticación
            response = requests.get(blob_url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            raise Exception(f"Error descargando desde Vercel Blob: {str(e)}")
    
    def download_to_file(self, blob_url: str, local_path: str) -> None:
        """
        Descarga un archivo desde Vercel Blob y lo guarda localmente
        
        Args:
            blob_url: URL completa del archivo en el blob
            local_path: Ruta local donde guardar el archivo
        """
        content = self.download_file(blob_url)
        with open(local_path, 'wb') as f:
            f.write(content)
    
    def download_to_memory(self, blob_url: str) -> io.BytesIO:
        """
        Descarga un archivo desde Vercel Blob a un BytesIO en memoria
        
        Args:
            blob_url: URL completa del archivo en el blob
            
        Returns:
            Objeto BytesIO con el contenido del archivo
        """
        content = self.download_file(blob_url)
        return io.BytesIO(content)


def get_blob_storage() -> VercelBlobStorage:
    """Factory function para obtener la instancia del storage"""
    return VercelBlobStorage()
