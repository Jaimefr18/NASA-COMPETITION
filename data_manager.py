import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import shutil

class DataManager:
    """
    Gerencia o armazenamento persistente dos dados de treinamento
    """
    
    def __init__(self, data_file="training_data.json", backup_file="training_data_backup.json"):
        self.data_file = data_file
        self.backup_file = backup_file
        self.training_data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Carrega dados do arquivo JSON"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ Dados carregados: {len(data)} registros")
                    return data
            else:
                print("📁 Arquivo de dados não encontrado. Criando novo.")
                return []
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            # Tenta carregar backup
            return self._load_backup()
    
    def _load_backup(self) -> List[Dict[str, Any]]:
        """Tenta carregar do arquivo de backup"""
        try:
            if os.path.exists(self.backup_file):
                with open(self.backup_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ Backup carregado: {len(data)} registros")
                    return data
            return []
        except Exception as e:
            print(f"❌ Erro ao carregar backup: {e}")
            return []
    
    def save_data(self) -> bool:
        """Salva dados no arquivo JSON"""
        try:
            # Cria backup primeiro
            if os.path.exists(self.data_file):
                shutil.copy2(self.data_file, self.backup_file)
            
            # Salva dados atuais
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Dados salvos: {len(self.training_data)} registros")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
            return False
    
    def add_training_data(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Adiciona dados de análise para treinamento (compatível com ML.py)
        """
        return self.add_analysis_data(analysis_data)
    
    def add_analysis_data(self, analysis_data: Dict[str, Any]) -> bool:
        """Adiciona nova análise aos dados"""
        try:
            # Adiciona timestamp único
            analysis_data['_id'] = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.training_data)}"
            analysis_data['_timestamp'] = datetime.now().isoformat()
            
            self.training_data.append(analysis_data)
            
            # Salva automaticamente após adicionar
            return self.save_data()
        except Exception as e:
            print(f"❌ Erro ao adicionar dados: {e}")
            return False
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Retorna todos os dados de treinamento"""
        return self.training_data
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos dados"""
        if not self.training_data:
            return {"total": 0, "sensors": {}, "period": {}}
        
        # Estatísticas por sensor
        sensors_count = {}
        for data in self.training_data:
            for sensor in data.get('sensors_used', []):
                sensors_count[sensor] = sensors_count.get(sensor, 0) + 1
        
        # Período dos dados
        dates = []
        for data in self.training_data:
            if 'period' in data:
                dates.append(data['period']['start'])
                dates.append(data['period']['end'])
        
        # Contagem de eventos por tipo
        event_types = {}
        for data in self.training_data:
            for event in data.get('events', []):
                event_type = event.get('type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_analyses": len(self.training_data),
            "sensors_count": sensors_count,
            "event_types": event_types,
            "data_period": {
                "earliest": min(dates) if dates else "N/A",
                "latest": max(dates) if dates else "N/A"
            },
            "total_events": sum(len(data.get('events', [])) for data in self.training_data)
        }
    
    def clear_data(self) -> bool:
        """Limpa todos os dados (cuidado!)"""
        try:
            self.training_data = []
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            if os.path.exists(self.backup_file):
                os.remove(self.backup_file)
            print("🗑️ Todos os dados foram limpos")
            return True
        except Exception as e:
            print(f"❌ Erro ao limpar dados: {e}")
            return False
    
    def export_to_csv(self, filename="training_data_export.csv") -> bool:
        """Exporta dados para CSV para análise"""
        try:
            if not self.training_data:
                print("❌ Nenhum dado para exportar")
                return False
            
            # Cria DataFrame com dados estruturados
            records = []
            for analysis in self.training_data:
                base_record = {
                    'analysis_id': analysis.get('_id', ''),
                    'timestamp': analysis.get('_timestamp', ''),
                    'lon': analysis.get('location', {}).get('lon', ''),
                    'lat': analysis.get('location', {}).get('lat', ''),
                    'start_date': analysis.get('period', {}).get('start', ''),
                    'end_date': analysis.get('period', {}).get('end', ''),
                    'sensors_used': ','.join(analysis.get('sensors_used', [])),
                    'total_events': len(analysis.get('events', []))
                }
                records.append(base_record)
            
            df = pd.DataFrame(records)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"📊 Dados exportados para {filename}")
            return True
        except Exception as e:
            print(f"❌ Erro ao exportar dados: {e}")
            return False
    
    def export_to_json(self, filename="training_data_export.json") -> bool:
        """Exporta dados para JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"📄 Dados exportados para {filename}")
            return True
        except Exception as e:
            print(f"❌ Erro ao exportar dados: {e}")
            return False
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Recupera uma análise específica pelo ID"""
        for analysis in self.training_data:
            if analysis.get('_id') == analysis_id:
                return analysis
        return None
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Remove uma análise específica"""
        try:
            original_length = len(self.training_data)
            self.training_data = [a for a in self.training_data if a.get('_id') != analysis_id]
            
            if len(self.training_data) < original_length:
                return self.save_data()
            return False
        except Exception as e:
            print(f"❌ Erro ao deletar análise: {e}")
            return False
    
    def get_analyses_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Filtra análises por período"""
        try:
            filtered_analyses = []
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            for analysis in self.training_data:
                analysis_date = datetime.strptime(analysis.get('_timestamp', ''), '%Y-%m-%dT%H:%M:%S.%f')
                if start <= analysis_date <= end:
                    filtered_analyses.append(analysis)
            
            return filtered_analyses
        except Exception as e:
            print(f"❌ Erro ao filtrar por data: {e}")
            return []
    
    def get_analyses_by_location(self, lon: float, lat: float, radius_km: float = 10) -> List[Dict[str, Any]]:
        """Filtra análises por localização (raio em km)"""
        try:
            filtered_analyses = []
            
            for analysis in self.training_data:
                analysis_lon = analysis.get('location', {}).get('lon', 0)
                analysis_lat = analysis.get('location', {}).get('lat', 0)
                
                # Calcula distância aproximada (fórmula de Haversine simplificada)
                distance = self._calculate_distance(lon, lat, analysis_lon, analysis_lat)
                
                if distance <= radius_km:
                    filtered_analyses.append(analysis)
            
            return filtered_analyses
        except Exception as e:
            print(f"❌ Erro ao filtrar por localização: {e}")
            return []
    
    def _calculate_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calcula distância aproximada entre dois pontos (km)"""
        # Fórmula de Haversine simplificada
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat/2)**2 + (lat1 + dlat/2) * (1 - (lat1 + dlat/2)) * (dlon/2)**2
        c = 2 * 6371 * (a ** 0.5)  # Raio da Terra ≈ 6371 km
        return c
    
    def backup_data(self, backup_dir: str = "backups") -> bool:
        """Cria um backup completo dos dados"""
        try:
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f"training_data_backup_{timestamp}.json")
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Backup criado em {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao criar backup: {e}")
            return False
    
    def restore_from_backup(self, backup_file: str) -> bool:
        """Restaura dados de um arquivo de backup"""
        try:
            if not os.path.exists(backup_file):
                print(f"❌ Arquivo de backup não encontrado: {backup_file}")
                return False
            
            with open(backup_file, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
            
            return self.save_data()
        except Exception as e:
            print(f"❌ Erro ao restaurar backup: {e}")
            return False