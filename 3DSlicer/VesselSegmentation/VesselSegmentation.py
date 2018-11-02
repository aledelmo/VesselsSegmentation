import os
import pickle
import shutil
import unittest
from builtins import range
# from contextlib import contextmanager
from subprocess import call

import ctk
import docker
import qt
import slicer

__author__ = 'Alessandro Delmonte'
__email__ = 'delmonte.ale92@gmail.com'


# def pipe(cmd, verbose=False, my_env=slicer.util.startupEnvironment()):
#     if verbose:
#         print('Processing command: ' + str(cmd))
#
#     slicer.app.processEvents()
#
#     return call(cmd, shell=True, stdin=None, stdout=None, stderr=None,
#                 executable=os.path.abspath(slicer.util.startupEnvironment()['SHELL']),
#                 env=my_env)
#
#
# @contextmanager
# def cd(newdir):
#     prevdir = os.getcwd()
#     os.chdir(os.path.expanduser(newdir))
#     try:
#         yield
#     finally:
#         os.chdir(prevdir)


class VesselSegmentation:
    def __init__(self, parent):
        parent.title = 'Vessels'
        parent.categories = ['IMAG2', 'Pelvic Segmentation']
        parent.dependencies = []
        parent.contributors = ['Alessandro Delmonte (IMAG2)']
        parent.helpText = '''INSERT HELP TEXT.'''
        parent.acknowledgementText = '''Module developed for 3DSlicer (<a>http://www.slicer.org</a>)'''

        self.parent = parent

        module_dir = os.path.dirname(self.parent.path)
        icon_path = os.path.join(module_dir, 'Resources', 'icon.png')
        if os.path.isfile(icon_path):
            parent.icon = qt.QIcon(icon_path)

        try:
            slicer.selfTests
        except AttributeError:
            slicer.selfTests = {}
        slicer.selfTests['VesselSegmentation'] = self.runTest

    def __repr__(self):
        return 'VesselSegmentation(parent={})'.format(self.parent)

    def __str__(self):
        return 'VesselSegmentation module initialization class.'

    @staticmethod
    def runTest():
        tester = VesselSegmentationTest()
        tester.runTest()


class VesselSegmentationWidget:
    def __init__(self, parent=None):
        self.moduleName = self.__class__.__name__
        if self.moduleName.endswith('Widget'):
            self.moduleName = self.moduleName[:-6]
        settings = qt.QSettings()
        try:
            self.developerMode = settings.value('Developer/DeveloperMode').lower() == 'true'
        except AttributeError:
            self.developerMode = settings.value('Developer/DeveloperMode') is True

        self.tmp = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'test_dataset', 'docker_infer')
        self.logic = VesselSegmentationLogic()

        if not parent:
            self.parent = slicer.qMRMLWidget()
            self.parent.setLayout(qt.QVBoxLayout())
            self.parent.setMRMLScene(slicer.mrmlScene)
        else:
            self.parent = parent
        self.layout = self.parent.layout()

        if not parent:
            self.setup()
            self.parent.show()

    def __repr__(self):
        return 'VesselSegmentationsWidget(parent={})'.format(self.parent)

    def __str__(self):
        return 'VesselSegmentation GUI class'

    def setup(self):
        vs_collapsible_button = ctk.ctkCollapsibleButton()
        vs_collapsible_button.text = 'Vessels Segmentation'

        self.layout.addWidget(vs_collapsible_button)

        vs_form_layout = qt.QFormLayout(vs_collapsible_button)
        vs_form_layout.setVerticalSpacing(13)

        self.volume_selector = slicer.qMRMLNodeComboBox()
        self.volume_selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.volume_selector.selectNodeUponCreation = True
        self.volume_selector.addEnabled = False
        self.volume_selector.removeEnabled = False
        self.volume_selector.noneEnabled = False
        self.volume_selector.showHidden = False
        self.volume_selector.renameEnabled = False
        self.volume_selector.showChildNodeTypes = False
        self.volume_selector.setMRMLScene(slicer.mrmlScene)

        self.volume_selector.connect('nodeActivated(vtkMRMLNode*)', self.on_volume_select)

        self.volume_node = self.volume_selector.currentNode()

        vs_form_layout.addRow('Reference Volume: ', self.volume_selector)

        self.output_selector = slicer.qMRMLNodeComboBox()
        self.output_selector.nodeTypes = ['vtkMRMLLabelMapVolumeNode']
        self.output_selector.selectNodeUponCreation = True
        self.output_selector.addEnabled = True
        self.output_selector.removeEnabled = False
        self.output_selector.noneEnabled = False
        self.output_selector.showHidden = False
        self.output_selector.renameEnabled = True
        self.output_selector.showChildNodeTypes = False
        self.output_selector.setMRMLScene(slicer.mrmlScene)

        self.output_selector.connect('currentNodeChanged(vtkMRMLNode*)', self.on_output_select)

        self.output_node = self.output_selector.currentNode()

        vs_form_layout.addRow('Output Seeds Label: ', self.output_selector)

        self.compute_button = qt.QPushButton('Compute')
        self.compute_button.enabled = True

        self.compute_button.connect('clicked(bool)', self.on_compute_button)
        vs_form_layout.addRow(self.compute_button)

        line = qt.QFrame()
        line.setFrameShape(qt.QFrame().HLine)
        line.setFrameShadow(qt.QFrame().Sunken)
        line.setStyleSheet("min-height: 24px")
        vs_form_layout.addRow(line)

        self.markups_selector = slicer.qSlicerSimpleMarkupsWidget()
        self.markups_selector.objectName = 'arteryFiducialsNodeSelector'
        self.markups_selector.setNodeBaseName("OriginSeedsArteries")
        self.markups_selector.defaultNodeColor = qt.QColor(200, 8, 21)
        self.markups_selector.maximumHeight = 150
        self.markups_selector.markupsSelectorComboBox().noneEnabled = False
        vs_form_layout.addRow("Arteries initialization:", self.markups_selector)
        self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                            self.markups_selector, 'setMRMLScene(vtkMRMLScene*)')

        self.markups_selector_vein = slicer.qSlicerSimpleMarkupsWidget()
        self.markups_selector_vein.objectName = 'veinFiducialsNodeSelector'
        self.markups_selector_vein.setNodeBaseName("OriginSeedsVeins")
        self.markups_selector_vein.defaultNodeColor = qt.QColor(65, 105, 225)
        self.markups_selector_vein.maximumHeight = 150
        self.markups_selector_vein.markupsSelectorComboBox().noneEnabled = False
        vs_form_layout.addRow("Veins initialization:", self.markups_selector_vein)
        self.parent.connect('mrmlSceneChanged(vtkMRMLScene*)',
                            self.markups_selector_vein, 'setMRMLScene(vtkMRMLScene*)')

        self.layout.addStretch(1)

        if self.developerMode:

            def createHLayout(elements):
                widget = qt.QWidget()
                rowLayout = qt.QHBoxLayout()
                widget.setLayout(rowLayout)
                for element in elements:
                    rowLayout.addWidget(element)
                return widget

            """Developer interface"""
            self.reloadCollapsibleButton = ctk.ctkCollapsibleButton()
            self.reloadCollapsibleButton.text = "Reload && Test"
            self.layout.addWidget(self.reloadCollapsibleButton)
            reloadFormLayout = qt.QFormLayout(self.reloadCollapsibleButton)

            self.reloadButton = qt.QPushButton("Reload")
            self.reloadButton.toolTip = "Reload this module."
            self.reloadButton.name = "ScriptedLoadableModuleTemplate Reload"
            self.reloadButton.connect('clicked()', self.onReload)

            self.reloadAndTestButton = qt.QPushButton("Reload and Test")
            self.reloadAndTestButton.toolTip = "Reload this module and then run the self tests."
            self.reloadAndTestButton.connect('clicked()', self.onReloadAndTest)

            self.editSourceButton = qt.QPushButton("Edit")
            self.editSourceButton.toolTip = "Edit the module's source code."
            self.editSourceButton.connect('clicked()', self.onEditSource)

            self.restartButton = qt.QPushButton("Restart Slicer")
            self.restartButton.toolTip = "Restart Slicer"
            self.restartButton.name = "ScriptedLoadableModuleTemplate Restart"
            self.restartButton.connect('clicked()', slicer.app.restart)

            reloadFormLayout.addWidget(
                createHLayout([self.reloadButton, self.reloadAndTestButton, self.editSourceButton, self.restartButton]))

    def on_volume_select(self):
        self.volume_node = self.volume_selector.currentNode()

    def on_output_select(self):
        self.output_node = self.output_selector.currentNode()

    def on_compute_button(self):

        if self.output_node and self.volume_node and self.markups_selector.currentNode() and \
                self.markups_selector_vein.currentNode():
            current_seeds_node = self.markups_selector.currentNode()
            art_init = []
            for n in range(current_seeds_node.GetNumberOfFiducials()):
                current = [0, 0, 0]
                current_seeds_node.GetNthFiducialPosition(n, current)
                art_init.append(current)

            art_path = os.path.join(self.tmp, 'arteries.pkl')
            with open(art_path, 'wb') as handle:
                pickle.dump(art_init, handle)

            current_seeds_node = self.markups_selector_vein.currentNode()
            vein_init = []
            for n in range(current_seeds_node.GetNumberOfFiducials()):
                current = [0, 0, 0]
                current_seeds_node.GetNthFiducialPosition(n, current)
                vein_init.append(current)

            vein_path = os.path.join(self.tmp, 'veins.pkl')
            with open(vein_path, 'wb') as handle:
                pickle.dump(vein_init, handle)

            properties = {'useCompression': 0}
            volume_path = os.path.join(self.tmp, 'reference.nii')
            slicer.util.saveNode(self.volume_node, volume_path, properties)

            segmentation_path = self.logic.segment()

            _, new_node = slicer.util.loadVolume(segmentation_path, returnNode=True)

            node_name = self.output_node.GetName()
            slicer.mrmlScene.RemoveNode(self.output_node)
            new_node.SetName(node_name)

            self.cleanup()

    def onReload(self):

        print('\n' * 2)
        print('-' * 30)
        print('Reloading module: ' + self.moduleName)
        print('-' * 30)
        print('\n' * 2)

        slicer.util.reloadScriptedModule(self.moduleName)

    def onReloadAndTest(self):
        try:
            self.onReload()
            test = slicer.selfTests[self.moduleName]
            test()
        except Exception:
            import traceback
            traceback.print_exc()
            errorMessage = "Reload and Test: Exception!\n\n" + "See Python Console for Stack Trace"
            slicer.util.errorDisplay(errorMessage)

    def onEditSource(self):
        filePath = slicer.util.modulePath(self.moduleName)
        qt.QDesktopServices.openUrl(qt.QUrl("file:///" + filePath, qt.QUrl.TolerantMode))

    def cleanup(self):
        for filename in os.listdir(self.tmp):
            if not filename.endswith('.md'):
                path = os.path.join(self.tmp, filename)
                if os.path.isfile(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)


class VesselSegmentationLogic:
    def __init__(self):
        self.temp_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
        self.client = docker.from_env(environment=slicer.util.startupEnvironment())

    def segment(self):

        self.client.containers.prune()
        self.client.images.build(path=self.temp_folder, dockerfile='Dockerfile',
                                 tag='test_docker_vessels')
        mounts = {
            os.path.join(self.temp_folder, 'test_dataset', 'docker_infer'): {'bind': '/test_docker', 'mode': 'rw'}}
        self.client.containers.run(image='test_docker_vessels', auto_remove=True, volumes=mounts, name='DeepVessel')

        # cmd = 'docker ps -aq --no-trunc -f status=exited | xargs docker rm'
        # pipe(cmd, True, self.my_env)
        #
        # cmd = 'docker build -t test_docker_vessels . && docker run -v ' + \
        #       os.path.join(self.temp_folder, 'test_dataset',
        #                    'docker_infer') + ':/test_docker --name DeepVessel test_docker_vessels '
        # pipe(cmd, True, self.my_env)

        output_path = os.path.join(self.temp_folder, 'test_dataset', 'docker_infer', 'seg-label.nii')

        return output_path


class VesselSegmentationTest(unittest.TestCase):

    def __init__(self):
        super(VesselSegmentationTest, self).__init__()

    def __repr__(self):
        return 'VesselSegmentation(). Derived from {}'.format(unittest.TestCase)

    def __str__(self):
        return 'VesselSegmentation test class'

    def runTest(self, scenario=None):
        pass
