#!/usr/bin/env python3
"""
Test Pub/Sub functionality
"""

import json
import time
from google.cloud import pubsub_v1
import os

def test_publisher(project_id, topic_name):
    """Test publishing messages to Pub/Sub"""
    
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)
    
    # Sample market data message
    sample_data = {
        "symbol": "AAPL",
        "timestamp": "2024-09-11T14:30:00Z",
        "close_price": 150.25,
        "volume": 1000000,
        "data_source": "test"
    }
    
    try:
        # Publish message
        message_data = json.dumps(sample_data).encode('utf-8')
        future = publisher.publish(topic_path, message_data)
        message_id = future.result()
        
        print(f" Published message {message_id} to {topic_name}")
        return True
        
    except Exception as e:
        print(f" Failed to publish to {topic_name}: {e}")
        return False

def test_subscriber(project_id, subscription_name, timeout=10):
    """Test subscribing to messages from Pub/Sub"""
    
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_name)
    
    received_messages = []
    
    def callback(message):
        print(f" Received message: {message.data.decode('utf-8')}")
        received_messages.append(message.data.decode('utf-8'))
        message.ack()
    
    try:
        print(f" Listening for messages on {subscription_name} for {timeout} seconds...")
        
        # Pull messages with timeout
        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        
        try:
            streaming_pull_future.result(timeout=timeout)
        except Exception:
            streaming_pull_future.cancel()
            
        print(f" Received {len(received_messages)} messages from {subscription_name}")
        return len(received_messages) > 0
        
    except Exception as e:
        print(f" Failed to subscribe to {subscription_name}: {e}")
        return False

def main():
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        print(" Please set GOOGLE_CLOUD_PROJECT environment variable")
        return
    
    print("  Testing Pub/Sub Setup...")
    print("=" * 50)
    
    # Test topics
    topics_to_test = [
        ("market-data-raw", "market-data-raw-sub"),
        ("market-data-processed", "market-data-processed-sub"),
        ("adversarial-alerts", "adversarial-alerts-sub")
    ]
    
    all_tests_passed = True
    
    for topic_name, subscription_name in topics_to_test:
        print(f"\n Testing {topic_name}...")
        
        # Test publishing
        pub_success = test_publisher(project_id, topic_name)
        
        # Small delay for message propagation
        time.sleep(2)
        
        # Test subscribing
        sub_success = test_subscriber(project_id, subscription_name, timeout=5)
        
        if pub_success and sub_success:
            print(f" {topic_name} test passed")
        else:
            print(f" {topic_name} test failed")
            all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print(" All Pub/Sub tests passed!")
    else:
        print("  Some Pub/Sub tests failed. Check your configuration.")

if __name__ == "__main__":
    main()