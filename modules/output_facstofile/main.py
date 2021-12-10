
import sys
import os
import argparse
from pathlib import Path

import json
import csv


if __name__ == '__main__':
    sys.path.append("..")
else:
    from modules.sendData import sendDataFrame


class MessageToFile:
    def __init__(self):
        self.counter = 0
        self.folderjson = Path("datajson")
        self.folderjson.mkdir(exist_ok=True)
        self.removefilesinfolder(self.folderjson)
        self.foldercsv = Path("datacsv")
        self.foldercsv.mkdir(exist_ok=True)
        # remove all existing files
        # shutil.rmtree(join(str(self.foldercsv), "*"))
        self.removefilesinfolder(self.foldercsv)

    # delete old files
    def removefilesinfolder(self, folder_path):
        for file in os.listdir(folder_path):
            file_path = Path(folder_path, file)
            try:
                file_path.unlink()
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def datajson(self, datajson):
        print(json.dumps(datajson, indent=4))
        # save data to JSON
        with open(Path(self.folderjson, "frame_{}.json".format(self.counter)), 'w') as outfile:
            json.dump(datajson, outfile)
        self.counter += 1

    def datacsv(self, key, data):
        # save data to CSV
        # print(data)

        # flatten dict
        au_dict = data.pop("au_r")
        pose_dict = data.pop("pose")
        # remove utc timestamp
        _ = data.pop("timestamp_utc", "")
        dict_flat = {**data, **pose_dict, **au_dict}
        print(dict_flat)

        with open(Path(self.foldercsv, "{}.csv".format(key)), 'a') as outfile:
            writer = csv.DictWriter(outfile, dict_flat.keys(), delimiter=',')
            if self.counter == 0:
                writer.writeheader()
            writer.writerow(dict_flat)
        #     writer = csv.writer(outfile, delimiter=',')
        #     if self.counter == 0:
        #         writer.writeheader()
        #     writer.writerow()

        self.counter += 1

    def stop(self):
        print("All messages received")


# client to message broker server
class sendingMessage(sendDataFrame):
    """Receives Head movement data; forward to output function"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.message_to_file = MessageToFile()

    async def sub(self):
        # keep listening to all incoming data
        while True:
            key, timestamp, data = await self.sub_socket.sub()
            print("Received message: {}".format([key, timestamp, data]))

            # check not finished; timestamp is empty (b'')
            if timestamp:
                # process data only
                if self.misc['file_format'] == "json":
                    self.message_to_file.datajson(data['au_r'])
                elif self.misc['file_format'] == "csv":
                    self.message_to_file.datacsv(key, data)

            # no more messages to be received
            else:
                print("No more messages to publish")
                self.message_to_json.stop()


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()

    # Sends the head movement data.
    parser.add_argument("--sub_ip", default=argparse.SUPPRESS,
                        help="IP (e.g. 192.168.x.x) of where to pub to; Default: 127.0.0.1 (local)")
    parser.add_argument("--sub_port", default="5571",
                        help="Port of where to pub to; Default: 5571")
    parser.add_argument("--sub_key", default=argparse.SUPPRESS,
                        help="Key for filtering message; Default: '' (all keys)")
    parser.add_argument("--sub_bind", default=False,
                        help="True: socket.bind() / False: socket.connect(); Default: False")

    # Module specific commandline arguments
    parser.add_argument("--file_format", default="json",
                        help="specific file format of how to store data;"
                             "json, csv; Default: json")

    args, leftovers = parser.parse_known_args()
    print("The following arguments are used: {}".format(args))
    print("The following arguments are ignored: {}\n".format(leftovers))

    # Start the data messaging class.
    data_messages = sendingMessage(**vars(args))
    # Start processing messages; give list of functions to call async
    data_messages.start([data_messages.sub])
