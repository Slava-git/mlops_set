import boto3
import json


def create_cloudwatch_dashboard():
    """Create CloudWatch dashboard for model monitoring"""
    
    cloudwatch = boto3.client('cloudwatch')
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["SageMaker/ModelMonitoring", "PredictionsPerMinute"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-west-2",
                    "title": "Кількість передбачень за хвилину",
                    "period": 60,
                    "stat": "Sum"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["SageMaker/ModelMonitoring", "ProcessingTime"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-west-2",
                    "title": "⏱️ Середній час обробки",
                    "period": 60,
                    "stat": "Average"
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 8,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["SageMaker/ModelMonitoring", "SuccessfulPredictions"],
                        [".", "FailedPredictions"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-west-2",
                    "title": "Успішні vs Невдалі передбачення",
                    "period": 300,
                    "stat": "Sum"
                }
            },
            {
                "type": "metric",
                "x": 8,
                "y": 6,
                "width": 8,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["SageMaker/ModelMonitoring", "InputLength"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-west-2",
                    "title": "📏 Довжина вхідних даних",
                    "period": 300,
                    "stat": "Average"
                }
            },
            {
                "type": "metric",
                "x": 16,
                "y": 6,
                "width": 8,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["SageMaker/ModelMonitoring", "ProcessingTime"]
                    ],
                    "view": "singleValue",
                    "region": "us-west-2",
                    "title": "⚡ Поточний час обробки",
                    "period": 300,
                    "stat": "Average"
                }
            }
        ]
    }
    
    try:
        cloudwatch.put_dashboard(
            DashboardName='LMSYS-Model-Monitoring',
            DashboardBody=json.dumps(dashboard_body)
        )
        
        print("Dashboard created!")
        print("URL: https://console.aws.amazon.com/cloudwatch/home?region=us-west-2#dashboards:name=LMSYS-Model-Monitoring")
        
    except Exception as e:
        print(f"Failed to create dashboard: {e}")

if __name__ == "__main__":
    create_cloudwatch_dashboard()