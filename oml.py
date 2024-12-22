import pika
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# RabbitMQ connection parameters
RABBITMQ_HOST = 'localhost'
INPUT_QUEUE = 'voltage_signals'
OUTPUT_QUEUE = 'anomaly_flags'

# Isolation Forest parameters
isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, warm_start=True, random_state=42)
scaler = StandardScaler()
initial_training_data = []
window_size = 100  # Initial training window size

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
channel = connection.channel()
channel.queue_declare(queue=INPUT_QUEUE)
channel.queue_declare(queue=OUTPUT_QUEUE)

# Function to publish anomalies to RabbitMQ
def publish_anomaly(data_point, anomaly_flag):
    message = {
        'voltage': data_point,
        'anomaly': anomaly_flag
    }
    channel.basic_publish(exchange='', routing_key=OUTPUT_QUEUE, body=json.dumps(message))

# Function to preprocess data
def preprocess(data_point):
    return scaler.transform(np.array(data_point).reshape(1, -1))

# Callback function for consuming RabbitMQ messages
def on_message(ch, method, properties, body):
    global initial_training_data
    data = json.loads(body)
    voltage = data['voltage']

    if len(initial_training_data) < window_size:
        # Collect initial training data
        initial_training_data.append([voltage])
        if len(initial_training_data) == window_size:
            # Initial training of the model
            initial_training_data = np.array(initial_training_data)
            scaler.fit(initial_training_data)
            scaled_data = scaler.transform(initial_training_data)
            isolation_forest.fit(scaled_data)
            print("Initial model training complete.")
    else:
        # Predict and update the model incrementally
        scaled_voltage = preprocess(voltage)
        anomaly_score = isolation_forest.decision_function(scaled_voltage)[0]
        anomaly_flag = anomaly_score < -0.1  # Define threshold

        # Publish the anomaly flag
        publish_anomaly(voltage, anomaly_flag)

        # Update the model incrementally
        isolation_forest.fit(np.vstack([isolation_forest.estimators_samples_, scaled_voltage]))

        print(f"Voltage: {voltage}, Anomaly: {anomaly_flag}, Score: {anomaly_score}")

# Start consuming messages
channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=on_message, auto_ack=True)
print("Waiting for messages. To exit, press CTRL+C")
channel.start_consuming()
