#!/usr/bin/env python3
"""
Monitor Progress of Ablation Study Data Generation
Displays real-time progress from the generation_progress.json file
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import curses

def format_time(seconds):
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def monitor_console(progress_file, refresh_interval=2):
    """Simple console monitoring without curses."""
    print("Monitoring ablation study progress...")
    print(f"Progress file: {progress_file}")
    print(f"Refresh interval: {refresh_interval}s")
    print("-" * 80)
    
    last_update = None
    
    try:
        while True:
            if not progress_file.exists():
                print("Waiting for progress file to be created...")
                time.sleep(refresh_interval)
                continue
            
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
            except:
                time.sleep(refresh_interval)
                continue
            
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Display header
            print("=" * 80)
            print("ABLATION STUDY PROGRESS MONITOR")
            print("=" * 80)
            print(f"Progress file: {progress_file}")
            print(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 80)
            
            # Display status
            status = progress.get('status', 'unknown')
            print(f"\nStatus: {status.upper()}")
            
            # Display timing
            if 'start_time' in progress:
                start_time = datetime.fromisoformat(progress['start_time'])
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"Elapsed time: {format_time(elapsed)}")
            
            if 'last_update' in progress:
                last_update_time = datetime.fromisoformat(progress['last_update'])
                since_update = (datetime.now() - last_update_time).total_seconds()
                print(f"Last update: {format_time(since_update)} ago")
            
            # Display progress bar
            percentage = progress.get('percentage', 0)
            bar_length = 50
            filled = int(bar_length * percentage / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\nProgress: [{bar}] {percentage}%")
            
            # Display ratios
            completed = progress.get('completed_ratios', [])
            failed = progress.get('failed_ratios', [])
            current = progress.get('current_ratio', None)
            
            print(f"\nCompleted: {len(completed)} ratios")
            if completed and len(completed) <= 5:
                for r in completed:
                    print(f"  ✓ {r}")
            elif completed:
                print(f"  ✓ {completed[-1]} (and {len(completed)-1} others)")
            
            if failed:
                print(f"\nFailed: {len(failed)} ratios")
                for r in failed[:5]:
                    print(f"  ✗ {r}")
                if len(failed) > 5:
                    print(f"  ... and {len(failed)-5} more")
            
            if current:
                print(f"\nCurrently processing: {current}")
            
            # Display recent messages
            messages = progress.get('messages', [])
            if messages:
                print(f"\nRecent activity:")
                for msg in messages[-5:]:
                    msg_time = datetime.fromisoformat(msg['time']).strftime('%H:%M:%S')
                    print(f"  [{msg_time}] {msg['message'][:70]}...")
            
            # Estimate time remaining
            if percentage > 0 and percentage < 100 and 'start_time' in progress:
                elapsed = (datetime.now() - start_time).total_seconds()
                estimated_total = elapsed / (percentage / 100)
                remaining = estimated_total - elapsed
                print(f"\nEstimated time remaining: {format_time(remaining)}")
            
            print("-" * 80)
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def monitor_curses(stdscr, progress_file, refresh_interval=2):
    """Enhanced monitoring with curses for better display."""
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)    # Non-blocking input
    stdscr.timeout(refresh_interval * 1000)  # Refresh in milliseconds
    
    # Define colors
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    
    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Check for quit
        key = stdscr.getch()
        if key == ord('q') or key == 27:  # q or ESC
            break
        
        if not progress_file.exists():
            stdscr.addstr(height//2, (width-30)//2, "Waiting for progress file...")
            stdscr.refresh()
            continue
        
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        except:
            continue
        
        row = 0
        
        # Header
        header = "ABLATION STUDY PROGRESS MONITOR"
        stdscr.addstr(row, (width - len(header))//2, header, curses.A_BOLD)
        row += 2
        
        # Status
        status = progress.get('status', 'unknown')
        status_color = curses.color_pair(1) if status == 'completed' else curses.color_pair(3)
        stdscr.addstr(row, 0, f"Status: ", curses.A_BOLD)
        stdscr.addstr(row, 8, status.upper(), status_color | curses.A_BOLD)
        row += 1
        
        # Timing
        if 'start_time' in progress:
            start_time = datetime.fromisoformat(progress['start_time'])
            elapsed = (datetime.now() - start_time).total_seconds()
            stdscr.addstr(row, 0, f"Elapsed: {format_time(elapsed)}")
            row += 1
        
        # Progress bar
        row += 1
        percentage = progress.get('percentage', 0)
        bar_width = min(width - 20, 60)
        filled = int(bar_width * percentage / 100)
        
        stdscr.addstr(row, 0, "Progress: [")
        for i in range(bar_width):
            if i < filled:
                stdscr.addstr("█", curses.color_pair(1))
            else:
                stdscr.addstr("░")
        stdscr.addstr(f"] {percentage}%")
        row += 2
        
        # Statistics
        completed = progress.get('completed_ratios', [])
        failed = progress.get('failed_ratios', [])
        current = progress.get('current_ratio', None)
        
        col1 = 0
        col2 = width // 2
        
        stdscr.addstr(row, col1, f"Completed: {len(completed)}", curses.color_pair(1) | curses.A_BOLD)
        if failed:
            stdscr.addstr(row, col2, f"Failed: {len(failed)}", curses.color_pair(2) | curses.A_BOLD)
        row += 1
        
        if current:
            stdscr.addstr(row, col1, f"Current: {current}", curses.color_pair(4))
            row += 1
        
        # Recent messages
        row += 1
        messages = progress.get('messages', [])
        if messages:
            stdscr.addstr(row, 0, "Recent activity:", curses.A_BOLD)
            row += 1
            
            max_messages = min(5, height - row - 3)
            for msg in messages[-max_messages:]:
                msg_time = datetime.fromisoformat(msg['time']).strftime('%H:%M:%S')
                msg_text = msg['message'][:width-12]
                stdscr.addstr(row, 2, f"[{msg_time}] {msg_text}")
                row += 1
        
        # Footer
        footer = "Press 'q' or ESC to quit"
        stdscr.addstr(height-1, (width - len(footer))//2, footer, curses.A_DIM)
        
        stdscr.refresh()

def main():
    parser = argparse.ArgumentParser(description="Monitor ablation study progress")
    parser.add_argument("--progress-file", default="../dev/ablation_study/generation_progress.json",
                      help="Path to progress file")
    parser.add_argument("--refresh", type=int, default=2,
                      help="Refresh interval in seconds")
    parser.add_argument("--simple", action="store_true",
                      help="Use simple console output instead of curses")
    parser.add_argument("--tail-log", type=str,
                      help="Also tail the specified log file")
    args = parser.parse_args()
    
    progress_file = Path(args.progress_file)
    
    if args.simple:
        monitor_console(progress_file, args.refresh)
    else:
        try:
            curses.wrapper(monitor_curses, progress_file, args.refresh)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        except Exception as e:
            print(f"Curses mode failed ({e}), falling back to simple mode...")
            monitor_console(progress_file, args.refresh)

if __name__ == "__main__":
    main()