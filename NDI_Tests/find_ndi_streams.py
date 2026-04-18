import NDIlib as ndi
import time

def find_iphone_source():
    find = ndi.find_create_v2()
    print("LOG: Starting deep scan for NDI|HX sources...")
    
    # We will loop for 15 seconds to give the iPhone time to respond
    for i in range(15):
        print(f"Scanning... ({15-i}s remaining)")
        ndi.find_wait_for_sources(find, 1000)
        sources = ndi.find_get_current_sources(find)
        
        if sources:
            print(f"\nSUCCESS! Found {len(sources)} sources:")
            for s in sources:
                name = s.ndi_name
                print(f" >> {name}")
                # Logic check: If it's an iPhone, it usually contains 'iPhone' or 'HX'
                if "iPhone" in name or "HX" in name:
                    print("!!! MATCH FOUND: Use this string in your RVM script !!!")
            
        time.sleep(0.5)

    ndi.find_destroy(find)
    print("\nLOG: Scan complete.")

if __name__ == "__main__":
    find_iphone_source()