"""A bunch of api calls functions that we used to talk to MTurk.
"""

from boto3 import Session
from datetime import datetime
from jinja2 import Environment
from jinja2 import FileSystemLoader

import json
import os
import xml


class EasyTurk(object):
    """Class that contains all the api calls to interface with MTurk.
    """

    def __init__(self, sandbox=True):
        """Constructor for EasyTurk.

        Args:
            sandbox: Whether we are launching on sandbox.
        """
        environments = {
                "production": {
                    "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
                    "preview": "https://www.mturk.com/mturk/preview"
                },
                "sandbox": {
                    "endpoint": "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
                    "preview": "https://workersandbox.mturk.com/mturk/preview"
                },
        }
        self.sandbox = sandbox
        env = (environments['sandbox'] if sandbox
               else environments["production"])
        self.session = Session(profile_name='mturk')
        self.mtc = self.session.client(
                service_name='mturk',
                region_name='us-east-1',
                endpoint_url=env['endpoint'],
        )

    def create_html_question(self, html, frame_height):
        head = ("<HTMLQuestion xmlns=\"http://mechanicalturk.amazonaws.com/"
                "AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd\">"
                "<HTMLContent><![CDATA[{}]]></HTMLContent><FrameHeight>")
        tail = "</FrameHeight></HTMLQuestion>"
        xml = head + str(frame_height) + tail
        return xml.format(html)

    def get_jinja_env(self):
        """Get a jinja2 Environment object that we can use to find templates.
        """
        dir_location = os.path.dirname(os.path.abspath(__file__))
        templates = os.path.join(dir_location, 'templates')
        return Environment(loader=FileSystemLoader(templates))

    def get_account_balance(self):
        """Retrieves the account balance.

        Returns:
            available account balance.
        """
        return self.mtc.get_account_balance()['AvailableBalance']

    def list_all_hits(self):
        hits = []
        curr_hits = self.mtc.list_hits()
        hits.extend(curr_hits["HITs"])
        while "NextToken" in curr_hits:
            curr_hits = self.mtc.list_hits( NextToken=curr_hits["NextToken"])
            hits.extend(curr_hits["HITs"])

        return hits

    def launch_hit(self, template_location, input_data, reward=0,
                   frame_height=9000, title=None, description=None,
                   keywords=None, duration=900, max_assignments=1,
                   country='US', hits_approved=10000, lifetime=1209600,
                   percent_approved=95, use_masters=False, qualification=None):
        """Launches a HIT.

        Make sure that none of the arguments are None.

        Returns:
            A hit_id.
        """
        if self.sandbox:
            percent_approved = 0
            hits_approved = 0
        hit_properties = {'Title': title,
                          'Description': description,
                          'Keywords': keywords,
                          'MaxAssignments': max_assignments,
                          'LifetimeInSeconds': lifetime,
                          'AssignmentDurationInSeconds': duration,
                          'QualificationRequirements': [
                              {
                                  'QualificationTypeId': '00000000000000000040',
                                  'Comparator': 'GreaterThanOrEqualTo',
                                  'IntegerValues': [hits_approved]
                              },
                              #{
                              #    'QualificationTypeId': '00000000000000000071',
                              #    'Comparator': 'EqualTo',
                              #    'LocaleValues': [
                              #         {'Country': country},
                              #    ],
                              #},
                              {
                                  'QualificationTypeId': '000000000000000000L0',
                                  'Comparator': 'GreaterThanOrEqualTo',
                                  'IntegerValues': [percent_approved],
                              }
                          ],
                          'Reward': str(reward)}
        if qualification:
            hit_properties["QualificationRequirements"] = [{
                'QualificationTypeId': qualification,
                'Comparator': 'Exists',
            }]

        if use_masters:
            hit_properties["QualificationRequirements"].append({
                'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
                'Comparator': 'Exists',
            })
        # Setup HTML Question.
        env = self.get_jinja_env()
        template = env.get_template(template_location)
        template_params = {'input': json.dumps(input_data)}
        html = template.render(template_params)
        html_question = self.create_html_question(html, frame_height)

        hit_properties['Question'] = html_question

        hit = self.mtc.create_hit(**hit_properties)
        return hit

    def _parse_response_from_assignment(self, assignment):
        """Parses out the worker's response from the assignment received.

        Args:
            assignment: A dictionary describing the assignment from boto.

        Returns:
            A Python object or list of the worker's response.
        """
        try:
            output_tree = xml.etree.ElementTree.fromstring(assignment['Answer'])
            output = json.loads(output_tree[0][1].text)
            return output
        except ValueError as e:
            print(e)
            return None

    def give_qualification(self, worker, qualification):
        self.mtc.associate_qualification_with_worker(
                QualificationTypeId=qualification,
                WorkerId=worker,
                IntegerValue=1,
                SendNotification=True
            )

    def get_results(self, hit_id, reject_on_fail=False):
        """Retrives the output of a hit if it has finished.

        Args:
            hit_id: The hit id of the HIT.
            reject_on_fail: If the hit returns unparsable answers,
                then reject it.

        Returns:
            A list of dictionaries with the following fields:
                - assignment_id
                - hit_id
                - worker_id
                - output
                - submit_time
            The number of dictionaries is equal to the number of assignments.
        """
        status = ['Approved', 'Submitted', 'Rejected']
        assignments = []
        try:
            tmp = self.mtc.list_assignments_for_hit(
                    HITId=hit_id, AssignmentStatuses=status)
            assignments.extend(tmp['Assignments'])
            while "NextToken" in tmp:
                tmp = self.mtc.list_assignments_for_hit(NextToken=tmp["NextToken"],
                    HITId=hit_id, AssignmentStatuses=status)
                assignments.extend(tmp['Assignments'])
        except Exception:
            return []
        results = []
        for a in assignments:
            output = self._parse_response_from_assignment(a)
            if output is not None:
                results.append({
                    'assignment_id': a['AssignmentId'],
                    'hit_id': hit_id,
                    'worker_id': a['WorkerId'],
                    'output': output,
                    'submit_time': a['SubmitTime'],
                })
            elif reject_on_fail:
                self.mtc.reject_assignment(
                    AssignmentId=a['AssignmentId'],
                    RequesterFeedback='Invalid results')
        return results

    def list_all_hits_qualification(self, qualification):
        hits = []
        curr_hits = self.mtc.list_hits_for_qualification_type(QualificationTypeId=qualification)
        hits.extend(curr_hits["HITs"])
        #print(curr_hits)
        print("NextToken" in curr_hits, curr_hits.keys())
        while "NextToken" in curr_hits:
            print(curr_hits["NextToken"], len(hits))
            curr_hits = self.mtc.list_hits_for_qualification_type(NextToken=curr_hits["NextToken"],
                                                                QualificationTypeId=qualification)
            hits.extend(curr_hits["HITs"])

        return hits

    def delete_hit(self, hit_id):
        """Disables a hit.

        Args:
            hit_id: The hit id to disable.

        Returns:
            A boolean indicating success.
        """
        try:
            self.mtc.delete_hit(HITId=hit_id)
            return True
        except Exception:
            try:
                self.mtc.update_expiration_for_hit(
                        HITId=hit_id,
                        ExpireAt=10)
                self.mtc.delete_hit(HITId=hit_id)
                return True
            except Exception as e:
                print(e)
                return False

    def approve_hit(self, hit_id, reject_on_fail=False,
                    override_rejection=False, message="Thanks for your work!"):
        """Approves a hit so that the worker can get paid.

        Args:
            hit_id: The hit id to disable.
            reject_on_fail: If the hit returns unparsable answers,
                then reject it.
            override_rejection: overrides a previous rejection if it exists.

        Returns:
           Tuple of approved and rejected assignment ids.
        """
        status = ['Approved', 'Submitted', 'Rejected']
        try:
            assignments = self.mtc.list_assignments_for_hit(
                    HITId=hit_id, AssignmentStatuses=status)
        except:
            return [], []
        approve_ids = []
        reject_ids = []
        for a in assignments['Assignments']:
            if a['AssignmentStatus'] == 'Submitted':
                output = self._parse_response_from_assignment(a)
                if output is not None:
                    approve_ids.append(a['AssignmentId'])
                elif reject_on_fail:
                    reject_ids.append(a['AssignmentId'])

        for assignment_id in approve_ids:
            self.mtc.approve_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback=message,
                    OverrideRejection=override_rejection)
        for assignment_id in reject_ids:
            self.mtc.reject_assignment(
                AssignmentId=assignment_id, RequesterFeedback='Invalid results')
        return approve_ids, reject_ids

    def reject_assignment(self, assignment_id):
        """Reject an assignment so that the worker can get paid.

        Args:
            assignment_id: An assignment id.

        Returns:
            A boolean indicating success.
        """
        a = self.mtc.get_assignment(AssignmentId=assignment_id)['Assignment']
        if a['AssignmentStatus'] == 'Submitted':
            self.mtc.reject_assignment(
                AssignmentId=assignment_id,
                RequesterFeedback='Invalid results')
            return True
        return False

    def approve_assignment(self, assignment_id, reject_on_fail=False,
                           override_rejection=False, message="Good job"):
        """Approves an assignment so that the worker can get paid.

        Args:
            assignment_id: An assignment id.
            reject_on_fail: If the hit returns unparsable answers,
                then reject it.
            override_rejection: overrides a previous rejection if it exists.

        Returns:
            A boolean indicating success.
        """
        a = self.mtc.get_assignment(AssignmentId=assignment_id)['Assignment']
        if a['AssignmentStatus'] == 'Submitted':
            output = self._parse_response_from_assignment(a)
            if output is not None:
                self.mtc.approve_assignment(
                        AssignmentId=assignment_id,
                        RequesterFeedback=message,
                        OverrideRejection=override_rejection)
                return True
            elif reject_on_fail:
                self.mtc.reject_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback='Invalid results')
                return False
        return False

    def show_hit_progress(self, hit_ids):
        """Show the progress of the hits.

        Args:
            hit_ids: A list of HIT ids.

        Returns:
            A dictionary from hit_id to a dictionary of completed
            and maximum assignments.
        """
        output = {}
        for hit_id in hit_ids:
            hit = self.mtc.get_hit(HITId=hit_id)
            assignments = self.mtc.list_assignments_for_hit(
                    HITId=hit_id, AssignmentStatuses=['Submitted'])
            completed = len(assignments['Assignments'])
            max_assignments = hit['HIT']['MaxAssignments']
            output[hit_id] = {'completed': completed,
                              'max_assignments': max_assignments}
        return output

    def list_hits(self):
        """Lists the HITs that have already been launched.

        Returns:
            A list of HITs.
        """
        hits = self.mtc.list_hits()['HITs']
        return hits

    def cancel_all_hits(self):
        hits = self.list_hits()
        [self.mtc.delete_hit(HITId=x["HITId"]) for x in hits]

    def create_hit_type(self,Reward, Title, Keywords, Description, QualificationRequirements,
                            AutoApprovalDelayInSeconds=604800,
                            AssignmentDurationInSeconds=1800,
                            ):

        return self.mtc.create_hit_type(Reward=str(Reward),
                                        Title=Title,
                                        Keywords=Keywords,
                                        Description=Description,
                                        QualificationRequirements=QualificationRequirements,
                                        AutoApprovalDelayInSeconds=AutoApprovalDelayInSeconds,
                                        AssignmentDurationInSeconds=AssignmentDurationInSeconds)

    def update_hit_type(self, hits, new_hit_type):
        if type(hits) is str:
            hits = [hits]

        if type(hits) is not list:
            raise Exception("Expecting list of hit types")

        updated_hits = []
        for hit in hits:
            self.mtc.update_hit_type_of_hit(HITId=hit, HITTypeId=new_hit_type)
            updated_hits.append(hit)
        return updated_hits

    def list_workers_with_qualification_type(self, qualification_type):
        workers = []

        data = self.mtc.list_workers_with_qualification_type(
            QualificationTypeId=qualification_type,
            Status='Granted',
        )

        workers.extend([x["WorkerId"] for x in data["Qualifications"]])
        while "NextToken" in data:
            data = self.mtc.list_workers_with_qualification_type(
                QualificationTypeId=qualification_type,
                Status='Granted',
                NextToken=data["NextToken"]
            )
            workers.extend([x["WorkerId"] for x in data["Qualifications"]])

        return workers

    def notify_workers(self, subject, message, workerIds):
        self.mtc.notify_workers(Subject=subject,
                                MessageText=message,
                                WorkerIds=workerIds)