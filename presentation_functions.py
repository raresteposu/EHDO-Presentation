import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def jährliche_bedarfe(results):
    demands = {
        "January": 
        {
            "power": float(results["Demands"]["monthly"]["power"]["January"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["January"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["January"])
        },
        "February":
        {
            "power": float(results["Demands"]["monthly"]["power"]["February"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["February"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["February"])
        },
        "March":
        {
            "power": float(results["Demands"]["monthly"]["power"]["March"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["March"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["March"])
        },
        "April":
        {
            "power": float(results["Demands"]["monthly"]["power"]["April"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["April"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["April"])
        },
        "May":
        {
            "power": float(results["Demands"]["monthly"]["power"]["May"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["May"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["May"])
        },
        "June":
        {
            "power": float(results["Demands"]["monthly"]["power"]["June"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["June"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["June"])
        },
        "July":
        {
            "power": float(results["Demands"]["monthly"]["power"]["July"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["July"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["July"])
        },
        "August":
        {
            "power": float(results["Demands"]["monthly"]["power"]["August"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["August"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["August"])
        },
        "September":
        {
            "power": float(results["Demands"]["monthly"]["power"]["September"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["September"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["September"])
        },
        "October":
        {
            "power": float(results["Demands"]["monthly"]["power"]["October"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["October"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["October"])
        },
        "November":
        {
            "power": float(results["Demands"]["monthly"]["power"]["November"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["November"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["November"])
        },
        "December":
        {
            "power": float(results["Demands"]["monthly"]["power"]["December"]),
            "heat": float(results["Demands"]["monthly"]["heat"]["December"]),
            "cool": float(results["Demands"]["monthly"]["cool"]["December"])
        }
    }

    # Create a figure with plots the power, heat and cool demands for each month

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    months = list(demands.keys())

    power = [demands[month]['power'] for month in months]
    heat = [demands[month]['heat'] for month in months]
    cool = [demands[month]['cool'] for month in months]

    for i in range(len(months)):
        months[i] = months[i][:3]

    ax[0].bar(months, power, color='blue')
    ax[0].set_title('Strombedarf pro Monat')
    ax[0].set_ylabel('Strom [kWh]')
    ax[0].set_xlabel('Monat')

    ax[1].bar(months, heat, color='red')
    ax[1].set_title('Wärmebedarf pro Monat')
    ax[1].set_ylabel('Wärme [kWh]')
    ax[1].set_xlabel('Monat')

    ax[2].bar(months, cool, color='green')
    ax[2].set_title('Kühlbedarf pro Monat')
    ax[2].set_ylabel('Kühlung [kWh]')
    ax[2].set_xlabel('Monat')

    plt.tight_layout()
    plt.show()

def plot_kapazitaten(results):
        
    capacities = { }

    for device in results["Devices"]:
        if device == "Grid":
            if results["Grid Flows"]["Maximum electricity export"] > 0:
                capacities.update({"Grid (Stromexport)": results["Grid Flows"]["Maximum electricity export"]})
            if results["Grid Flows"]["Maximum gas import"] > 0:
                capacities.update({"Grid (Gasexport)": results["Grid Flows"]["Maximum gas export"]})
            if results["Grid Flows"]["Maximum heat export"] > 0:
                capacities.update({"Grid (Wärmeexport)": results["Grid Flows"]["Maximum heat export"]})
        else:
            capacities.update({device: results["Devices"][device]["Capacity"]})

    fig, ax = plt.subplots(figsize=(10, 5))

    # Show the capacities of the devices, including the number on the top of the bars

    devices = list(capacities.keys())
    values = list(capacities.values())

    ax.bar(devices, values, color='blue')
    ax.set_title('Kapazität der Anlagen')
    ax.set_ylabel('Kapazität [kW]')
    ax.set_xlabel('Gerät')

    for i in range(len(devices)):
        ax.text(i, values[i], str(values[i]), ha='center', va='bottom')

def plot_mittelwert(available_devices, building, devices, types, sub_types, hour_range, day_range):
    
    if devices[0] not in available_devices:
        raise ValueError(f"Analge {devices[0]} nicht verfügbar")

    from itertools import product

    types.append('/')
    sub_types.append('/')
    df  = pd.read_csv(f'../results/hourly_values_{building}.csv')

    # Initialize a plot
    plt.figure(figsize=(10, 5))

    # Generate all combinations of device, type_, and sub_type
    for device, type_, sub_type in product(devices, types, sub_types):
        filtered_df = df[
            (df['Device'] == device) &
            (df['Type'] == type_) &
            (df['Sub-Type'] == sub_type) &
            (df['Hour'] >= hour_range[0]) & (df['Hour'] <= hour_range[1]) &
            (df['Day'] >= day_range[0]) & (df['Day'] <= day_range[1])
        ]

        # Initialize a vector of size 365 with NaN values
        median_values = np.full(365, np.nan)

        # Calculate the median value for each day in the specified range
        for day in range(day_range[0], day_range[1]+1):
            daily_values = filtered_df[filtered_df['Day'] == day]['Value']
            if not daily_values.empty:
                median_values[day] = daily_values.median()

        # Check if the median_values contain any non-NaN values
        if not np.isnan(median_values).all():
            # Plot the median values
            
            plt.plot(median_values, label=f'{device} - {type_} - {sub_type}')

    # Add labels and legend
    plt.xlabel('Tag')
    plt.ylabel('Medianwert')
    plt.legend(loc='upper right')
    plt.show()

def plot_kosten_art(results):
    
    type_costs = { 
        "Investment": 0,
        "O&M": 0,
        "Nachfrageabhängig": 0,
    }

    if results["Total Costs"]["CO2 Tax"] > 0.1:
        type_costs["CO2 Steuer"] = results["Total Costs"]["CO2 Tax"]

    # Fill the type_costs dictionary

    type_costs["Investment"] = results["Total Costs"]["Total device costs"]["Total investment costs"]
    type_costs["O&M"] = results["Total Costs"]["Total device costs"]["Total O&M costs"]
    type_costs["Nachfrageabhängig"] = results["Total Costs"]["Total device costs"]["Total demand related costs"]

    if results["Grid Flows"]["Total electricity import"] != 0: 
        type_costs["Strom von Grid"] = results["Total Costs"]["Supply costs"]["Electricity"]["Supply costs"]
    if results["Grid Flows"]["Total heat import"] != 0 :
        type_costs["Wärme von Grid"] = results["Total Costs"]["Supply costs"]["Heat"]["Supply costs"]
    if results["Grid Flows"]["Total gas import"] != 0:
        type_costs["Gas von Grid"] = results["Total Costs"]["Supply costs"]["Gas"]["Supply costs"]
    if results["Grid Flows"]["Total biomass import"] != 0:
        type_costs["Biomasse von Grid"] = results["Total Costs"]["Supply costs"]["Biomass"]

    # Create the labels and values for the pie chart

    labels_types = list(type_costs.keys())
    values_types = list(type_costs.values())

    # Create the pie chart

    fig, ax = plt.subplots()

    wedges, texts = ax.pie(values_types, labels=None, autopct=None, textprops=dict(color="w"))


    # Add the legend with the labels and values

    ax.legend(wedges, [f"{label}: {value} €" for label, value in zip(labels_types, values_types)],
                title="Typen",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1))

    fig.tight_layout()

    plt.show()


def plot_erzeugte_energie(results):
    demands = results["Demands"]["sum"].keys()
    for demand in list(demands):
        if results["Demands"]["sum"][demand] == 0:
            del results["Demands"]["sum"][demand]
    devices = results["Devices"]

    # Add the Grid to the devices

    devices["Grid"] = {
        "Generated": {
            "power": results["Grid Flows"]["Total electricity import"],
            "heat": results["Grid Flows"]["Total heat import"]
        }
    }

    generated_data = {
        "heat": {device: [] for device in devices if device not in ["TES", "BAT"]},
        "cool": {device: [] for device in devices if device not in ["TES", "BAT"]},
        "power": {device: [] for device in devices if device not in ["TES", "BAT"]}
    }

    for device in devices:
        if isinstance(devices[device]["Generated"], dict):
            for demand in demands:
                if demand in devices[device]["Generated"]:
                    generated_data[demand][device].append(devices[device]["Generated"][demand])
        else:
            if device in ["HP","STC", "BOI", "EB", "BBOI"]: 
                generated_data["heat"][device].append(devices[device]["Generated"])
                generated_data["cool"][device].append(0)
                generated_data["power"][device].append(0)
            elif device in ["AC", "CC"]:
                generated_data["heat"][device].append(0)
                generated_data["cool"][device].append(devices[device]["Generated"])
                generated_data["power"][device].append(0)
            elif device in ["PV", "WT", "WAT", "CHP", "BCHP", "WCHP", "FC"]:
                generated_data["heat"][device].append(0)
                generated_data["cool"][device].append(0)
                generated_data["power"][device].append(devices[device]["Generated"])


    for demand in generated_data:
        for device in generated_data[demand]:
            generated_data[demand][device] = sum(generated_data[demand][device])


    def print_as_text (generated_data, results, devices):
        print(f"🔥 Wärmebedarf: {results['Demands']['sum']['heat']} kW")
        for device in devices:
            if generated_data["heat"][device] != 0:
                print(f"    - {device}: {generated_data['heat'][device]} kWh ({round(generated_data['heat'][device] / results['Demands']['sum']['heat'] * 100,1)}% des Gesamtwärmebedarfs)")


        print(f"⚡⚡ Strombedarf: {results['Demands']['sum']['power']} kW")
        for device in devices:
            if generated_data["power"][device] != 0:
                print(f"    - {device}: {generated_data['power'][device]} kWh ({round(generated_data['power'][device] / results['Demands']['sum']['power'] * 100,1)}% des Gesamtstrombedarfs)")

        print(f"❄️ Kühlbedarf: ", results["Demands"]["sum"]["cool"])
        for device in devices:
            if generated_data["cool"][device] != 0:
                print(f"    - {device}: {generated_data['cool'][device]} kWh ({round(generated_data['cool'][device] / results['Demands']['sum']['cool'] * 100,1)}% des Gesamtkühlbedarfs)")
        
    def create_diagramm_for_erzeugte_energie(generated_data, demands, devices):

        labels = list(devices.keys()) if devices[device]["Generated"] != 0 else list(devices.keys())[:-1]
        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots()

        bars = []
        for i, demand in enumerate(demands):
            values = [generated_data[demand][device] for device in labels]
            bars.append(ax.bar(x + i*width, values, width, label=demand))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Anlagen')
        ax.set_ylabel('Erzeugte Energie/Kühlung/Wärme [kWh]')
        ax.set_title('Erzeugung (nach Gerät)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.legend()

        # Add labels to each bar
        for bar in bars:
            ax.bar_label(bar)

        # Modify the legend, to show Wämrme instead of heat
        handles, labels = ax.get_legend_handles_labels()
        for i, label in enumerate(labels):
            if label == "heat":
                labels[i] = "Wärme"
            elif label == "cool":
                labels[i] = "Kühlung"
            elif label == "power":
                labels[i] = "Strom"
        ax.legend(handles, labels)




        fig.tight_layout()

        plt.show()

    # make a copy of devices and remove the TES and BAT devices

    devices_new= devices.copy()
    devices_new.pop("TES", None)
    devices_new.pop("BAT", None)


    create_diagramm_for_erzeugte_energie(generated_data, demands, devices_new)

def kosten_nach_gerät(results):
    
    func = lambda pct, allvals: "{:.1f} % \n{:d} €".format(pct, int(pct/100.*sum(allvals)))

    # Create the cost by device chart 

    devices = results["Devices"]
    devices_costs = { }

    # Fill the devices_costs dictionary

    for device in devices:
        if device == "Grid":
            devices_costs.update({"Grid" : results["Total Costs"]["Supply costs"]["Total Costs"]})
        elif devices[device]["Costs"]["Total"] != 0:
            devices_costs.update({device: devices[device]["Costs"]["Total"]})

    # Create the labels and values for the pie chart
    labels_devices = list(devices_costs.keys())
    values_devices = [devices_costs[device] for device in labels_devices]

    fig, ax = plt.subplots()
    wedges, texts = ax.pie(values_devices, labels=None, autopct=None, textprops=dict(color="w"))

    # Add the legend with the labels and values
    ax.legend(wedges, [f"{label}: {value} €/a" for label, value in zip(labels_devices, values_devices)],
            title="Anlagen",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.show()