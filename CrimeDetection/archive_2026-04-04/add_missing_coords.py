import pandas as pd

# Read existing coordinates
coords = pd.read_csv('district_coordinates.csv')

# New entries for missing districts with accurate Malaysian geographic coordinates
new_entries = [
    ['Sabah', 'Kota Marudu', 6.2000, 117.1000],
    ['Sabah', 'Kudat', 6.9167, 118.7500],
    ['Sabah', 'Ranau', 5.2833, 118.1667],
    ['Sabah', 'Semporna', 4.4833, 118.6000],
    ['Sabah', 'Sipitang', 4.9167, 115.7500],
    ['Sabah', 'Tenom', 5.1333, 115.6667],
    ['Sabah', 'Tuaran', 6.3333, 117.1667],
    ['Sabah', 'W.P. Labuan', 5.2833, 115.2333],
    ['Sarawak', 'Julau', 2.3333, 112.5333],
    ['Sarawak', 'Kanowit', 2.0333, 111.8667],
    ['Sarawak', 'Kota Samarahan', 1.4667, 110.5000],
    ['Sarawak', 'Lawas', 4.8333, 115.4167],
    ['Sarawak', 'Limbang', 4.7667, 115.4833],
    ['Sarawak', 'Lubok Antu', 0.9667, 111.2333],
    ['Sarawak', 'Marudi', 3.2000, 113.0667],
    ['Sarawak', 'Matu Daro', 2.0667, 111.6667],
    ['Sarawak', 'Meradong', 1.8167, 111.8333],
    ['Sarawak', 'Padawan', 1.3667, 110.4000],
    ['Sarawak', 'Saratok', 1.3333, 111.7000],
    ['Sarawak', 'Sarikei', 2.1333, 111.5167],
    ['Sarawak', 'Serian', 1.2667, 110.8500],
    ['Sarawak', 'Tatau', 3.0000, 112.2333],
    ['Selangor', 'Serdang', 2.7333, 101.9500],
    ['Selangor', 'Sg. Buloh', 3.2000, 101.5500],
    ['Selangor', 'Subang Jaya', 3.0667, 101.5667],
    ['Selangor', 'Sungai Buloh', 3.2000, 101.5500],
    ['W.P. Kuala Lumpur', 'All', 3.1390, 101.6869],
    ['W.P. Kuala Lumpur', 'Brickfields', 3.1167, 101.6833],
    ['W.P. Kuala Lumpur', 'Cheras', 3.0667, 101.7500],
    ['W.P. Kuala Lumpur', 'Dang Wangi', 3.1500, 101.7000],
    ['W.P. Kuala Lumpur', 'Sentul', 3.1833, 101.6667],
    ['W.P. Kuala Lumpur', 'W.P. Putrajaya', 2.9269, 101.6964],
    ['W.P. Kuala Lumpur', 'Wangsa Maju', 3.1667, 101.7167],
]

new_df = pd.DataFrame(new_entries, columns=['state', 'district', 'latitude', 'longitude'])
coords = pd.concat([coords, new_df], ignore_index=True)
coords.to_csv('district_coordinates.csv', index=False)
print(f'✅ Added {len(new_entries)} missing districts. Total: {len(coords)}')
