function export_utah_forge_table(input_path, output_csv_path, output_json_path, requested_columns_csv)
if nargin < 4
    requested_columns_csv = 'time,tau,v_int,d_int,mu';
end

loaded = load(input_path);
names = fieldnames(loaded);
if isempty(names)
    error('No variables found in %s.', input_path);
end

table_name = names{1};
table_value = loaded.(table_name);
if ~istable(table_value)
    error('Expected a MATLAB table in %s but found %s.', input_path, class(table_value));
end

requested_columns = string(strsplit(requested_columns_csv, ','));
requested_columns = strtrim(requested_columns);
available_columns = string(table_value.Properties.VariableNames);
selected_columns = requested_columns(ismember(requested_columns, available_columns));
if isempty(selected_columns)
    error('None of the requested columns were found in %s.', input_path);
end

selected_table = table_value(:, cellstr(selected_columns));
writetable(selected_table, output_csv_path);

summary = struct();
summary.source_file = input_path;
summary.table_name = table_name;
summary.variable_names = cellstr(available_columns);
summary.extracted_columns = cellstr(selected_columns);
summary.table_shape = [height(table_value), width(table_value)];
summary.csv_path = output_csv_path;

json_text = jsonencode(summary, PrettyPrint=true);
fid = fopen(output_json_path, 'w');
if fid < 0
    error('Could not open %s for writing.', output_json_path);
end
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', json_text);
end
