var PythonShell = require('python-shell');

var options = {
	mode: 'text',
	pythonPath: '',
	pythonOptions: ['-u'],
	scriptPath: '',
	args: ['value1', 'value2', 'value3']
};

function upload(file_path, callback) {
	PythonShell.run('./google_drive/google-drive-upload.py file_path', options, function(err, results) {
		if (err) throw err;

		console.log('results : %j', results);
		console.log('results.length : %d', results.length);
		console.log('prediction : %s', results[results.length - 1]);
		callback(results)
	});
}

exports.upload = upload;
