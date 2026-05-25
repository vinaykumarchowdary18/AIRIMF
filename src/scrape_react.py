import requests
import pandas as pd
import time
import sys

# ⚠️ PUT YOUR REAL GITHUB TOKEN HERE ⚠️
GITHUB_TOKEN = 'github_pat_*************************************'
HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

REPO_OWNER = 'facebook'
REPO_NAME = 'react'
BASE_URL = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues'

def fetch_issues(state='closed', labels='Bug', per_page=100, max_pages=20):
    """
    state: 'closed' (we want resolved issues)
    labels: 'Bug' (capital B – React uses this exact label)
    per_page: max 100 per page
    max_pages: how many pages to try
    """
    all_issues = []
    for page in range(1, max_pages + 1):
        params = {
            'state': state,
            'labels': labels,
            'per_page': per_page,
            'page': page,
            'sort': 'created',
            'direction': 'desc'
        }
        resp = requests.get(BASE_URL, headers=HEADERS, params=params)
        # Debug output
        print(f"Request URL: {resp.url}")
        print(f"Status code: {resp.status_code}")

        if resp.status_code != 200:
            print(f"Error on page {page}: {resp.status_code}")
            break

        issues = resp.json()
        print(f"Items on this page: {len(issues)}")

        if not issues:
            print("No more issues. Stopping pagination.")
            break

        # Keep only real issues (skip pull requests)
        for issue in issues:
            if 'pull_request' not in issue:
                all_issues.append({
                    'Issue_ID': issue['number'],
                    'Title': issue['title'],
                    'Body': issue['body'] if issue['body'] else '',
                    'State': issue['state'],
                    'Created_At': issue['created_at'],
                    'Closed_At': issue['closed_at'],
                    'Comments': issue['comments'],
                    'Body_Length': len(issue['body']) if issue['body'] else 0
                })

        print(f"Total valid issues so far: {len(all_issues)}")
        time.sleep(1)   # be respectful to the API

    return all_issues

print("🚀 Starting scrape of facebook/react (label='Bug')...")
data = fetch_issues(max_pages=20)

if not data:
    print("❌ No issues collected. Maybe the label 'Bug' isn't right.")
    print("   Trying fallback: fetching all closed issues and filtering locally...")
    # Fallback: get all closed issues and filter for any label containing 'bug' (case-insensitive)
    data = fetch_issues(state='closed', labels=None, max_pages=30)
    bug_data = []
    for issue in data:
        # issue at this point is a dict without 'labels' because we didn't fetch them?
        # We need to modify fallback to actually fetch label names.
        pass

# A better fallback that fetches label info:
if not data:
    # Fetch without label filter, but this time get labels as well
    data_all = []
    for page in range(1, 31):
        params = {
            'state': 'closed',
            'per_page': 100,
            'page': page,
            'sort': 'created',
            'direction': 'desc'
        }
        resp = requests.get(BASE_URL, headers=HEADERS, params=params)
        if resp.status_code != 200:
            break
        issues = resp.json()
        if not issues:
            break
        for issue in issues:
            if 'pull_request' not in issue:
                # check labels
                label_names = [lbl['name'].lower() for lbl in issue.get('labels', [])]
                if any('bug' in ln for ln in label_names):
                    data_all.append({
                        'Issue_ID': issue['number'],
                        'Title': issue['title'],
                        'Body': issue['body'] if issue['body'] else '',
                        'State': issue['state'],
                        'Created_At': issue['created_at'],
                        'Closed_At': issue['closed_at'],
                        'Comments': issue['comments'],
                        'Body_Length': len(issue['body']) if issue['body'] else 0
                    })
        print(f"Fallback page {page}: collected {len(data_all)} bug issues so far")
        time.sleep(1)
    data = data_all

# Now process data
if not data:
    print("❌ Still no issues. Check your token or network.")
    sys.exit()

df = pd.DataFrame(data)
print(f"✅ Total issues scraped: {len(df)}")

# Save raw CSV (includes Body text)
df.to_csv('react_bugs_raw.csv', index=False)
print("Saved raw data to react_bugs_raw.csv")

# Prepare the pipeline-friendly CSV (same columns as vscode_bugs_datasets.csv)
pipeline_df = df[['Issue_ID', 'Title', 'Body_Length', 'Created_At', 'Closed_At', 'Comments']]
pipeline_df.to_csv('react_bugs_dataset.csv', index=False)
print("Saved pipeline-ready data to react_bugs_dataset.csv")
print("✅ Scrape complete.")
