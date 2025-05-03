#!/usr/bin/env python
import os
import csv
import time
import requests
import sys
import json

# Constants
MAJESTIC_MILLION_URL = "https://downloads.majestic.com/majestic_million.csv"
LOCAL_CSV_PATH = "majestic_million.csv"
API_ENDPOINT = "http://localhost:5000/api/process"
PROGRESS_FILE = "scraper_progress.json"

def download_majestic_million():
    """Download the Majestic Million CSV file if it doesn't exist locally"""
    if os.path.exists(LOCAL_CSV_PATH):
        print(f"Majestic Million data already exists at {LOCAL_CSV_PATH}")
        return True
    
    print(f"Downloading Majestic Million data from {MAJESTIC_MILLION_URL}...")
    try:
        response = requests.get(MAJESTIC_MILLION_URL)
        response.raise_for_status()  # Raise exception for non-200 responses
        
        with open(LOCAL_CSV_PATH, 'wb') as f:
            f.write(response.content)
        
        print(f"Download complete! Saved to {LOCAL_CSV_PATH}")
        return True
    except Exception as e:
        print(f"Error downloading Majestic Million data: {e}")
        return False

def get_agreement_type():
    """Ask user if they want to scrape pp, tos, or both"""
    print("\nSelect the agreement type to process:")
    print("1. TOS (Terms of Service)")
    print("2. PP (Privacy Policy)")
    print("3. Both TOS and PP")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            return "tos"
        elif choice == "2":
            return "pp"
        elif choice == "3":
            return "both"
        else:
            print("Invalid input. Please enter a number from 1 to 3.")

def get_success_target():
    """Ask user for target number of successful scrapes"""
    while True:
        try:
            target = int(input("Enter your target number of successful scrapes: "))
            if target > 0:
                return target
            print("Target must be greater than 0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_start_rank(agreement_type):
    """Get starting rank, offering to continue from previous run"""
    last_rank = get_last_processed_rank(agreement_type)
    
    if last_rank:
        continue_last = input(f"Current progress: rank {last_rank} for {agreement_type.upper()}. Continue from next rank? (y/n): ")
        if continue_last.lower().strip() in ['y', 'yes']:
            return last_rank + 1
        
        # If user doesn't want to continue from saved progress, ask for a custom rank
        while True:
            try:
                rank = int(input("Enter the starting rank (1-1,000,000): "))
                if 1 <= rank <= 1000000:
                    return rank
                print("Rank must be between 1 and 1,000,000.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        # No saved progress, default to rank 1 without asking
        return 1

def get_last_processed_rank(agreement_type):
    """Get the last processed rank from the progress file"""
    if not os.path.exists(PROGRESS_FILE):
        return None
    
    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
            return progress.get(agreement_type, {}).get('rank')
    except Exception:
        return None

def get_progress_stats(agreement_type):
    """Get the full progress stats from the progress file"""
    if not os.path.exists(PROGRESS_FILE):
        return {'rank': 0, 'successes': 0}
    
    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
            return progress.get(agreement_type, {'rank': 0, 'successes': 0})
    except Exception:
        return {'rank': 0, 'successes': 0}

def save_progress(agreement_type, rank, successes):
    """Save the current progress to the progress file"""
    progress = {}
    
    # Load existing progress if available
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
        except:
            pass
    
    # Update with the current stats
    progress[agreement_type] = {
        'rank': rank,
        'successes': successes
    }
    
    # Save back to file
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def display_progress_stats(agreement_type, current_rank, successes, success_target):
    """Display detailed progress statistics"""
    remaining = max(0, success_target - successes)
    print("\n" + "="*50)
    print(f"PROGRESS REPORT - {agreement_type.upper()}")
    print(f"Current rank: {current_rank}")
    print(f"Successful scrapes: {successes}")
    print(f"Remaining to reach target: {remaining}")
    print(f"Progress: {(successes/success_target*100):.1f}% complete")
    print("="*50)

def process_domains(agreement_type, start_rank, success_target):
    """Process domains from the CSV starting at the given rank"""
    try:
        # Get existing progress stats
        progress_stats = get_progress_stats(agreement_type)
        successes = progress_stats.get('successes', 0)
        
        with open(LOCAL_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Skip rows until we reach the start rank
            for row in reader:
                current_rank = int(row['GlobalRank'])
                
                if current_rank >= start_rank:
                    domain = row['Domain']
                    success = process_domain(domain, agreement_type)
                    
                    # Update success count
                    if success:
                        successes += 1
                    
                    # Save progress after each domain
                    save_progress(agreement_type, current_rank, successes)
                    
                    # Display current progress stats
                    display_progress_stats(agreement_type, current_rank, successes, success_target)
                    
                    # Check if we've reached the success target
                    if successes >= success_target:
                        print(f"\n🎉 SUCCESS TARGET REACHED! {successes} successful scrapes completed.")
                        return
                
                # No longer asking if user wants to continue after every 5 domains
                # Script will run continuously until success target is reached
                
    except Exception as e:
        print(f"Error processing CSV: {e}")

def process_domain(domain, agreement_type):
    """Call the API endpoint to process a single domain"""
    if agreement_type == "both":
        # Process both TOS and PP
        print(f"\nProcessing {domain} for both TOS and PP...")
        
        # Process TOS first
        tos_success = process_single_agreement(domain, "tos")
        
        # Process PP next
        pp_success = process_single_agreement(domain, "pp")
        
        # Summary of results
        if tos_success and pp_success:
            print(f"✅ Success: {domain} - Both agreements processed successfully")
            return True
        elif tos_success:
            print(f"⚠️ Partial success: {domain} - Only TOS processed successfully")
            return True  # Count as success if at least one agreement was processed
        elif pp_success:
            print(f"⚠️ Partial success: {domain} - Only PP processed successfully")
            return True  # Count as success if at least one agreement was processed
        else:
            print(f"❌ Failed: {domain} - Neither agreement processed successfully")
            return False
    else:
        # Process single agreement type
        return process_single_agreement(domain, agreement_type)

def process_single_agreement(domain, agreement_type):
    """Process a single agreement type for a domain, return success status"""
    print(f"Processing {domain} for {agreement_type.upper()}...")
    payload = {
        "url": domain,
        "agreement_type": agreement_type
    }
    
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        result = response.json()
        
        if response.status_code == 200 and result.get('success'):
            print(f"✅ Success: {domain} - {agreement_type.upper()}")
            print(f"Summary: {result.get('summary_25', '')[:100]}...")
            return True
        else:
            print(f"❌ Failed: {domain} - {agreement_type.upper()}")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"Error calling API for {domain} - {agreement_type.upper()}: {e}")
        return False
    finally:
        # Sleep to avoid overwhelming the server
        time.sleep(2)

def continue_processing():
    """Ask user if they want to continue processing domains"""
    response = input("\nContinue processing? (y/n): ").lower().strip()
    return response == 'y' or response == 'yes'

def main():
    # Download Majestic Million data if needed
    if not download_majestic_million():
        print("Cannot continue without Majestic Million data.")
        return
    
    # Get user preferences
    agreement_type = get_agreement_type()
    success_target = get_success_target()
    start_rank = get_start_rank(agreement_type)
    
    # Process domains
    print(f"\nStarting to process domains from rank {start_rank} for {agreement_type.upper()}...")
    print(f"Target: {success_target} successful scrapes")
    process_domains(agreement_type, start_rank, success_target)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 