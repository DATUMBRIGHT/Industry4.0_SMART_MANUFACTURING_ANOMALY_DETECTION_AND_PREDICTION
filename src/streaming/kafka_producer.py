import json
import time
from typing import Dict, Any
from kafka import KafkaProducer


class ManufacturingKafkaProducer:
    """
    Kafka Producer for manufacturing IoT data with anomalies
    """
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8')
        )
        
        
        # Define topics for different data types
        self.topics = {
            "raw_sensor_data": "manufacturing.sensor.raw",
            "anomaly_alerts": "manufacturing.anomaly.alerts",
            "maintenance_alerts": "manufacturing.maintenance.alerts"
        }
    
    def send_sensor_data(self, record: Dict[str, Any]) -> None:
        """
        Send sensor data to Kafka topic
        """
        # Use machine_id as key for partitioning
        key = str(record.get("machine_id", "unknown"))
        
        # Send to raw sensor data topic
        self.producer.send(
            topic=self.topics["raw_sensor_data"],
            key=key,
            value=record
        )
        
        # Send anomaly alerts if anomaly detected
        if record.get("anomaly_flag") == 1:
            anomaly_alert = {
                "timestamp": record["timestamp"],
                "machine_id": record["machine_id"],
                "anomaly_type": self._detect_anomaly_type(record),
                "severity": self._calculate_severity(record),
                "sensor_values": {
                    "temperature": record.get("temperature"),
                    "vibration": record.get("vibration"),
                    "humidity": record.get("humidity"),
                    "pressure": record.get("pressure"),
                    "energy_consumption": record.get("energy_consumption")
                }
            }
            
            self.producer.send(
                topic=self.topics["anomaly_alerts"],
                key=key,
                value=anomaly_alert
            )
        
        # Send maintenance alerts if maintenance required
        if record.get("maintenance_required") == 1:
            maintenance_alert = {
                "timestamp": record["timestamp"],
                "machine_id": record["machine_id"],
                "predicted_remaining_life": record.get("predicted_remaining_life"),
                "downtime_risk": record.get("downtime_risk"),
                "failure_type": record.get("failure_type")
            }
            
            self.producer.send(
                topic=self.topics["maintenance_alerts"],
                key=key,
                value=maintenance_alert
            )
    
    def _detect_anomaly_type(self, record: Dict[str, Any]) -> str:
        """
        Detect the type of anomaly based on sensor values
        """
        temp = record.get("temperature", 0)
        vib = record.get("vibration", 0)
        energy = record.get("energy_consumption", 0)
        
        if temp > 100:  # High temperature threshold
            return "temperature_anomaly"
        elif vib > 60:  # High vibration threshold
            return "vibration_anomaly"
        elif energy > 5:  # High energy consumption
            return "energy_anomaly"
        else:
            return "general_anomaly"
    
    def _calculate_severity(self, record: Dict[str, Any]) -> str:
        """
        Calculate anomaly severity
        """
        downtime_risk = record.get("downtime_risk", 0)
        remaining_life = record.get("predicted_remaining_life", 100)
        
        if downtime_risk > 0.8 or remaining_life < 20:
            return "critical"
        elif downtime_risk > 0.5 or remaining_life < 50:
            return "warning"
        else:
            return "info"
    
    def stream_to_kafka(self, interval_seconds: float = 1.0, max_records: int = None):
        """
        Stream data from simulator to Kafka
        """
        print(f"Starting Kafka producer...")
        print(f"Topics: {self.topics}")
        
        record_count = 0
        
        try:
            for record in self.simulator.stream_data(interval_seconds, max_records):
                self.send_sensor_data(record)
                record_count += 1
                
                if record_count % 100 == 0:
                    print(f"Sent {record_count} records to Kafka")
                
                # Flush periodically to ensure delivery
                if record_count % 10 == 0:
                    self.producer.flush()
                    
        except KeyboardInterrupt:
            print(f"\nStopped streaming. Total records sent: {record_count}")
        finally:
            self.producer.close()
    
    def send_batch(self, batch_size: int = 100):
        """
        Send a batch of records to Kafka
        """
        print(f"Sending batch of {batch_size} records...")
        
        for i, record in enumerate(self.simulator.stream_data(interval_seconds=0.1, max_records=batch_size)):
            self.send_sensor_data(record)
            
            if (i + 1) % 10 == 0:
                print(f"Sent {i + 1}/{batch_size} records")
        
        self.producer.flush()
        print("Batch sent successfully!")

# Example usage
if __name__ == "__main__":
    producer = ManufacturingKafkaProducer()
    
    # Send a small batch for testing
    producer.send_batch(batch_size=50)
    
    # Uncomment to stream continuously
    # producer.stream_to_kafka(interval_seconds=1.0) 