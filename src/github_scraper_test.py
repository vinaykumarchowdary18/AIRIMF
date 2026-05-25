import requests
import pandas as pd
import time

# --- CONFIGURATION ---
GITHUB_TOKEN = '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'
HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

# We will test by pulling data from the VS Code repository
REPO_OWNER = 'microsoft'
REPO_NAME = 'vscode'
ISSUES_URL = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues'

def test_fetch_issues():
    print(f"Attempting to fetch data from {REPO_OWNER}/{REPO_NAME}...")
    
    # Parameters: We want closed issues (resolved) that are labeled as bugs
    params = {
        'state': 'closed',
        'labels': 'bug',
        'per_page': 10, # Keeping it small for the test
        'page': 1
    }
    
    response = requests.get(ISSUES_URL, headers=HEADERS, params=params)
    
    if response.status_code == 200:
        issues = response.json()
        data = []
        
        for issue in issues:
            # We only want actual issues, not pull requests for this dataset
            if 'pull_request' not in issue:
                data.append({
                    'Issue_ID': issue['number'],
                    'Title': issue['title'],
                    'State': issue['state'],
                    'Created_At': issue['created_at'],
                    'Closed_At': issue['closed_at'],
                    'Comments': issue['comments']
                })
                
        df = pd.DataFrame(data)
        print("\nSuccess! Here is a sample of the extracted data:\n")
        print(df.head())
        
        # Save to a CSV to verify file writing permissions
        df.to_csv('test_project_risks.csv', index=False)
        print("\nData saved to 'test_project_risks.csv'.")
        
    else:
        print(f"Error fetching data. Status Code: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    test_fetch_issues()