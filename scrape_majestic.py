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
            if agreement_type == "both":
                # For "both" mode, return the minimum rank between TOS and PP
                # to ensure we don't miss any domains
                tos_rank = progress.get('tos', {}).get('rank', 0)
                pp_rank = progress.get('pp', {}).get('rank', 0)
                return min(tos_rank, pp_rank) if tos_rank > 0 and pp_rank > 0 else max(tos_rank, pp_rank)
            else:
                return progress.get(agreement_type, {}).get('rank')
    except Exception:
        return None

def get_progress_stats(agreement_type):
    """Get the full progress stats from the progress file"""
    if not os.path.exists(PROGRESS_FILE):
        if agreement_type == "both":
            return {
                'tos': {'rank': 0, 'successes': 0},
                'pp': {'rank': 0, 'successes': 0}
            }
        else:
            return {'rank': 0, 'successes': 0}
    
    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
            if agreement_type == "both":
                # Return separate stats for TOS and PP
                tos_stats = progress.get('tos', {'rank': 0, 'successes': 0})
                pp_stats = progress.get('pp', {'rank': 0, 'successes': 0})
                return {'tos': tos_stats, 'pp': pp_stats}
            else:
                return progress.get(agreement_type, {'rank': 0, 'successes': 0})
    except Exception:
        if agreement_type == "both":
            return {
                'tos': {'rank': 0, 'successes': 0},
                'pp': {'rank': 0, 'successes': 0}
            }
        else:
            return {'rank': 0, 'successes': 0}

def save_progress(agreement_type, rank=None, successes=None, tos_rank=None, tos_successes=None, pp_rank=None, pp_successes=None):
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
    if agreement_type == "both":
        # Update TOS stats
        if 'tos' not in progress:
            progress['tos'] = {'rank': 0, 'successes': 0}
        
        if tos_rank is not None:
            progress['tos']['rank'] = tos_rank
        if tos_successes is not None:
            progress['tos']['successes'] = tos_successes
            
        # Update PP stats
        if 'pp' not in progress:
            progress['pp'] = {'rank': 0, 'successes': 0}
            
        if pp_rank is not None:
            progress['pp']['rank'] = pp_rank
        if pp_successes is not None:
            progress['pp']['successes'] = pp_successes
    else:
        progress[agreement_type] = {
            'rank': rank or 0,
            'successes': successes or 0
        }
    
    # Save back to file
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def display_progress_stats(agreement_type, current_rank, success_target, successes=None, tos_successes=None, pp_successes=None, current_focus=None):
    """Display detailed progress statistics"""
    print("\n" + "="*50)
    print(f"PROGRESS REPORT - {agreement_type.upper()}")
    
    if agreement_type == "both":
        # Get progress stats for current display
        progress_stats = get_progress_stats(agreement_type)
        tos_stats = progress_stats.get('tos', {})
        pp_stats = progress_stats.get('pp', {})
        
        tos_current_rank = tos_stats.get('rank', 0)
        pp_current_rank = pp_stats.get('rank', 0)
        tos_current_successes = tos_successes if tos_successes is not None else tos_stats.get('successes', 0)
        pp_current_successes = pp_successes if pp_successes is not None else pp_stats.get('successes', 0)
        
        tos_remaining = max(0, success_target - tos_current_successes)
        pp_remaining = max(0, success_target - pp_current_successes)
        
        print(f"Current rank: {current_rank}")
        print(f"TOS rank: {tos_current_rank}, PP rank: {pp_current_rank}")
        print(f"CURRENT FOCUS: {current_focus.upper() if current_focus else 'TOS'}")
        print(f"TOS successful scrapes: {tos_current_successes}")
        print(f"PP successful scrapes: {pp_current_successes}")
        print(f"TOS remaining to reach target: {tos_remaining}")
        print(f"PP remaining to reach target: {pp_remaining}")
        print(f"TOS progress: {(tos_current_successes/success_target*100):.1f}% complete")
        print(f"PP progress: {(pp_current_successes/success_target*100):.1f}% complete")
    else:
        remaining = max(0, success_target - successes)
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
        
        if agreement_type == "both":
            tos_stats = progress_stats.get('tos', {})
            pp_stats = progress_stats.get('pp', {})
            tos_successes = tos_stats.get('successes', 0) 
            pp_successes = pp_stats.get('successes', 0)
            
            # Determine which document type to focus on
            current_focus = "tos" if tos_successes < success_target else "pp"
            print(f"\nFocusing on {current_focus.upper()} first until target of {success_target} is reached.")
        else:
            successes = progress_stats.get('successes', 0)
        
        with open(LOCAL_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Skip rows until we reach the start rank
            for row in reader:
                current_rank = int(row['GlobalRank'])
                
                if current_rank >= start_rank:
                    domain = row['Domain']
                    
                    if agreement_type == "both":
                        # Check if we need to switch focus
                        if current_focus == "tos" and tos_successes >= success_target:
                            current_focus = "pp"
                            print(f"\n🎉 TOS TARGET REACHED! Switching focus to PP. Need {success_target - pp_successes} more successes.")
                        elif current_focus == "pp" and pp_successes >= success_target:
                            print(f"\n🎉 SUCCESS TARGET REACHED FOR BOTH! TOS: {tos_successes}, PP: {pp_successes} successful scrapes completed.")
                            return
                        
                        # Process focused document type only
                        success = process_single_agreement(domain, current_focus)
                        
                        # Update success counts
                        if success:
                            if current_focus == "tos":
                                tos_successes += 1
                            else:
                                pp_successes += 1
                        
                        # Save progress after each domain
                        if current_focus == "tos":
                            save_progress(agreement_type, tos_rank=current_rank, tos_successes=tos_successes,
                                         pp_rank=pp_stats.get('rank', 0), pp_successes=pp_successes)
                        else:
                            save_progress(agreement_type, tos_rank=tos_stats.get('rank', 0), tos_successes=tos_successes,
                                         pp_rank=current_rank, pp_successes=pp_successes)
                        
                        # Display current progress stats
                        display_progress_stats(agreement_type, current_rank, success_target, 
                                             tos_successes=tos_successes, pp_successes=pp_successes,
                                             current_focus=current_focus)
                        
                        # Check if we've reached both success targets
                        if tos_successes >= success_target and pp_successes >= success_target:
                            print(f"\n🎉 SUCCESS TARGET REACHED FOR BOTH! TOS: {tos_successes}, PP: {pp_successes} successful scrapes completed.")
                            return
                    else:
                        # Process single agreement type
                        success = process_domain(domain, agreement_type)
                        
                        # Update success count
                        if success:
                            successes += 1
                        
                        # Save progress after each domain
                        save_progress(agreement_type, rank=current_rank, successes=successes)
                        
                        # Display current progress stats
                        display_progress_stats(agreement_type, current_rank, success_target, successes=successes)
                        
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
        elif tos_success:
            print(f"⚠️ Partial success: {domain} - Only TOS processed successfully")
        elif pp_success:
            print(f"⚠️ Partial success: {domain} - Only PP processed successfully")
        else:
            print(f"❌ Failed: {domain} - Neither agreement processed successfully")
        
        # Return separate success statuses for TOS and PP
        return tos_success, pp_success
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