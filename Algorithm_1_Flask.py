from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Corrected function to format the available meeting times
def format_meeting_times_commas_corrected_fixed(row, time_slot_columns):
    meeting_times = []
    for col in time_slot_columns:
        if not pd.isna(row[col]):
            time_info = col.replace("What times are you available? [", "").replace("]", "")
            hour = ''.join(filter(str.isdigit, time_info))
            am_pm = ''.join(filter(str.isalpha, time_info))

            if hour.isdigit():
                hour = int(hour)
            else:
                continue

            days = row[col].split(';')

            if hour == 12:
                end_hour = 1
            else:
                end_hour = hour + 1
            end_am_pm = am_pm if hour < 11 else ('PM' if am_pm == 'AM' else 'AM')

            for day in days:
                meeting_time = f"{day} {hour} {am_pm} - {end_hour} {end_am_pm}"
                meeting_times.append(meeting_time)

    return ", ".join(sort_meeting_times(meeting_times))

# Helper function to sort meeting times
def sort_meeting_times(meeting_times):
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sorting_key = lambda s: (days_order.index(s.split()[0]), int(s.split()[1]))
    return sorted(meeting_times, key=sorting_key)

# Function to prepare data from CSV files with priority system
def prepare_data_final(df, student_type, priority_options, non_priority_options):
    schedule_columns = [col for col in df.columns if
                        "What times are you available?" in col and 'Score' not in col and 'Feedback' not in col]

    modality_col = "If you are interested in a Conversation Partnership, please select your preference for meeting." if student_type == 'domestic' \
        else "Please select your preference for meetings with your conversation partner."

    cleaned_df = pd.DataFrame({
        "Name": df["First Name"] + ' ' + df["Last Name"],
        "Email": df["TAMU Email"],
        "UIN": df["UIN"],
        "Modality": df[modality_col],
        "Schedule": df[schedule_columns].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)
    })

    cleaned_df = cleaned_df[cleaned_df["Modality"] != "Not interested in a Conversation Partnership"]
    # Adjust modality for domestic students volunteering for credit
    if student_type == 'domestic':
        credit_col = "Are you volunteering to receive credit for a course or certificate program?"
        if credit_col in df.columns:
            df.loc[df[credit_col] == 'Yes', modality_col] = "Via Zoom video conferencing"
        cleaned_df['Modality'] = df[modality_col]
    teaching_responsibilities_col = next(
        (col for col in df.columns if "What teaching responsibilities do you currently have?" in col), None)

    if teaching_responsibilities_col and student_type == 'international':
        cleaned_df['Priority'] = df[teaching_responsibilities_col].apply(
            lambda x: 1 if any(option in str(x) for option in priority_options) else 2
        )
    else:
        cleaned_df['Priority'] = 2

    cleaned_df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    cleaned_df = cleaned_df.sort_values(by=['Priority', 'Timestamp'])
    return cleaned_df



# Function to check schedule overlap
def improved_schedule_overlap(formatted_times1, formatted_times2):
    set1 = set(formatted_times1.split(', ')) if formatted_times1 else set()
    set2 = set(formatted_times2.split(', ')) if formatted_times2 else set()
    overlap = set1.intersection(set2)
    return ', '.join(overlap), bool(overlap)


# Updated matching function
def improved_match_students_v2(domestic, international):
    matches = []
    unmatched_domestic = domestic.copy()
    international = international.sort_values(by=['Priority', 'Timestamp'])

    for int_index, int_row in international.iterrows():
        potential_matches = []

        for dom_index, dom_row in unmatched_domestic.iterrows():
            overlap, has_overlap = improved_schedule_overlap(int_row['Formatted Meeting Times'],
                                                             dom_row['Formatted Meeting Times'])
            if has_overlap and (int_row['Modality'] == dom_row['Modality'] or 'No preference' in [int_row['Modality'],
                                                                                                  dom_row['Modality']]):
                potential_matches.append((dom_index, overlap))

        if potential_matches:
            potential_matches.sort(key=lambda x: unmatched_domestic.loc[x[0], 'Timestamp'])
            chosen_match, chosen_overlap = potential_matches[0]
            matches.append((int_index, chosen_match, chosen_overlap))
            unmatched_domestic.drop(chosen_match, inplace=True)

    unmatched_international_indices = set(international.index) - set([m[0] for m in matches])
    unmatched_international = international.loc[unmatched_international_indices]

    return matches, unmatched_domestic, unmatched_international

def choose_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)
    return file_path

# Process the CSV files (adapted from your Tkinter functions)
def process_csv_files(ns_file_path, nns_file_path):
    if ns_file_path and nns_file_path:
        # Read the CSV files
        domestic_df_new = pd.read_csv(ns_file_path)
        international_df_new = pd.read_csv(nns_file_path)

        # Define priority and non-priority options
        priority_options = ['Lab', 'Recitation Section', 'Office Hours', 'Lecturing', 'Instructor of record']
        non_priority_options = ['None', 'Grading']

        # Prepare data with priority system
        domestic_cleaned_df_final = prepare_data_final(domestic_df_new, 'domestic', priority_options,
                                                       non_priority_options)
        international_cleaned_df_final = prepare_data_final(international_df_new, 'international', priority_options,
                                                            non_priority_options)

        # Extract time slot columns and apply formatting function
        international_time_slots = [col for col in international_df_new.columns if
                                    'What times are you available?' in col and 'Score' not in col]
        domestic_time_slots = [col for col in domestic_df_new.columns if
                               'What times are you available?' in col and 'Score' not in col]
        international_df_new['Formatted Meeting Times'] = international_df_new.apply(
            lambda row: format_meeting_times_commas_corrected_fixed(row, international_time_slots), axis=1
        )
        domestic_df_new['Formatted Meeting Times'] = domestic_df_new.apply(
            lambda row: format_meeting_times_commas_corrected_fixed(row, domestic_time_slots), axis=1
        )

        # Merge formatted times into cleaned dataframes
        international_cleaned_df_final['Formatted Meeting Times'] = international_df_new['Formatted Meeting Times']
        domestic_cleaned_df_final['Formatted Meeting Times'] = domestic_df_new['Formatted Meeting Times']

        # Perform matching
        matched_pairs, unmatched_domestic, unmatched_international = improved_match_students_v2(
            domestic_cleaned_df_final, international_cleaned_df_final)

        # Create the final matched DataFrame
        matched_df = pd.DataFrame({
            "NNS - First and Last Name": international_cleaned_df_final.loc[
                [m[0] for m in matched_pairs], "Name"].values,
            "NNS - Email": international_cleaned_df_final.loc[[m[0] for m in matched_pairs], "Email"].values,
            "NNS - UIN": international_cleaned_df_final.loc[[m[0] for m in matched_pairs], "UIN"].values,
            "NNS - Modality": international_cleaned_df_final.loc[[m[0] for m in matched_pairs], "Modality"].values,
            "NS - First and Last Name": domestic_cleaned_df_final.loc[[m[1] for m in matched_pairs], "Name"].values,
            "NS - Email": domestic_cleaned_df_final.loc[[m[1] for m in matched_pairs], "Email"].values,
            "NS - UIN": domestic_cleaned_df_final.loc[[m[1] for m in matched_pairs], "UIN"].values,
            "NS - Modality": domestic_cleaned_df_final.loc[[m[1] for m in matched_pairs], "Modality"].values,
            "Modality": [domestic_cleaned_df_final.loc[m[1], "Modality"] if domestic_cleaned_df_final.loc[
                                                                                m[1], "Modality"] != "No preference"
                         else international_cleaned_df_final.loc[m[0], "Modality"] for m in matched_pairs],
            "Available Meeting Times": [m[2] for m in matched_pairs]
        })

        # Prepare unmatched individuals DataFrame
        unmatched_domestic['Type'] = 'domestic'
        unmatched_international['Type'] = 'international'
        unmatched_combined = pd.concat([unmatched_domestic, unmatched_international], ignore_index=True)
        unmatched_detailed = unmatched_combined[['Name', 'UIN', 'Email', 'Modality', 'Type']]

        # Write results to Excel
        output_stream = BytesIO()
        with pd.ExcelWriter(output_stream) as writer:
            matched_df.to_excel(writer, sheet_name='Matched Pairs', index=False)
            unmatched_detailed.to_excel(writer, sheet_name='Unmatched Individuals', index=False)
        output_stream.seek(0)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        ns_file = request.files.get('ns_file')
        nns_file = request.files.get('nns_file')

        if not ns_file or not nns_file:
            flash('Missing file(s)')
            return redirect(request.url)

        ns_filename = secure_filename(ns_file.filename)
        nns_filename = secure_filename(nns_file.filename)

        ns_file_path = os.path.join(app.config['UPLOAD_FOLDER'], ns_filename)
        nns_file_path = os.path.join(app.config['UPLOAD_FOLDER'], nns_filename)

        ns_file.save(ns_file_path)
        nns_file.save(nns_file_path)

        processed_file = process_csv_files(ns_file_path, nns_file_path)

        return send_file(processed_file, as_attachment=True, download_name='Processed_Matches.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
