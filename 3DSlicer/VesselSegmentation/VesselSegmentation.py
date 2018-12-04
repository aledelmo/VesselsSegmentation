from __future__ import division

import os
import pickle
import shutil
import sys
import unittest
from builtins import range

import ctk
import docker
import qt
import slicer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from training.patches import get_patches

__author__ = 'Alessandro Delmonte'
__email__ = 'delmonte.ale92@gmail.com'


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
        icon_path = os.path.join(module_dir, 'Resources', 'icon.jpg')
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

        patch_collapsible_button = ctk.ctkCollapsibleButton()
        patch_collapsible_button.text = 'Dataset Preparation'
        patch_collapsible_button.collapsed = True

        self.layout.addWidget(patch_collapsible_button)

        patch_form_layout = qt.QFormLayout(patch_collapsible_button)

        self.label_selector = slicer.qMRMLNodeComboBox()
        self.label_selector.nodeTypes = ['vtkMRMLLabelMapVolumeNode']
        self.label_selector.selectNodeUponCreation = True
        self.label_selector.addEnabled = False
        self.label_selector.removeEnabled = False
        self.label_selector.noneEnabled = False
        self.label_selector.showHidden = False
        self.label_selector.renameEnabled = False
        self.label_selector.showChildNodeTypes = False
        self.label_selector.setMRMLScene(slicer.mrmlScene)

        self.label_selector.connect('currentNodeChanged(vtkMRMLNode*)', self.on_label_select)

        self.label_node = self.label_selector.currentNode()

        patch_form_layout.addRow('Label Map: ', self.label_selector)

        self.reference_selector = slicer.qMRMLNodeComboBox()
        self.reference_selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.reference_selector.selectNodeUponCreation = True
        self.reference_selector.addEnabled = False
        self.reference_selector.removeEnabled = False
        self.reference_selector.noneEnabled = False
        self.reference_selector.showHidden = False
        self.reference_selector.renameEnabled = False
        self.reference_selector.showChildNodeTypes = False
        self.reference_selector.setMRMLScene(slicer.mrmlScene)

        self.reference_selector.connect('currentNodeChanged(vtkMRMLNode*)', self.on_ref_select)

        self.ref_node = self.reference_selector.currentNode()

        patch_form_layout.addRow('Reference T2: ', self.reference_selector)

        self.dialog_folder_button = ctk.ctkDirectoryButton()
        self.dir = None
        self.dialog_folder_button.connect('directoryChanged(const QString&)', self.on_apply_dialog_folder_button)

        patch_form_layout.addRow('Patch Directory: ', self.dialog_folder_button)

        grid_layout = qt.QGridLayout()
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setColumnStretch(2, 1)
        grid_layout.setColumnStretch(3, 1)
        grid_layout.setColumnStretch(4, 1)

        self.radio_standard = qt.QRadioButton('Legacy LabelMap (16, 17)')
        self.radio_imag = qt.QRadioButton('IMAG2 format (43, 44)')
        self.radio_imag.setChecked(True)
        grid_layout.addWidget(self.radio_standard, 0, 2, 0)
        grid_layout.addWidget(self.radio_imag, 0, 4, 0)

        patch_form_layout.addRow(grid_layout)

        self.patch_button = qt.QPushButton('Save Patches')
        self.patch_button.enabled = True

        self.patch_button.connect('clicked(bool)', self.on_patch_button)
        patch_form_layout.addRow(self.patch_button)

        self.clean_button = qt.QPushButton('Clean Folder')
        self.clean_button.enabled = True

        self.clean_button.connect('clicked(bool)', self.on_clean_button)
        patch_form_layout.addRow(self.clean_button)

        train_collapsible_button = ctk.ctkCollapsibleButton()
        train_collapsible_button.text = 'Net Training'
        train_collapsible_button.collapsed = True

        self.layout.addWidget(train_collapsible_button)

        train_form_layout = qt.QFormLayout(train_collapsible_button)

        # self.train_folder_button = ctk.ctkDirectoryButton()
        # self.traindir = None
        # self.train_folder_button.connect('directoryChanged(const QString&)', self.on_apply_train_folder_button)

        # train_form_layout.addRow('Training Set Directory: ', self.train_folder_button)

        # stylesheet = """QPushButton {
        #                         background-color: rgba(128,128,128,50);
        #                         border: 1px solid black; border-radius: 5px;
        #                         }"""
        self.train_button = qt.QPushButton('Train')
        self.train_button.enabled = True
        # self.train_button.setStyleSheet(stylesheet)

        self.train_button.connect('clicked(bool)', self.on_train_button)
        train_form_layout.addRow(self.train_button)

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

    def on_label_select(self):
        self.label_node = self.label_selector.currentNode()

    def on_ref_select(self):
        self.ref_node = self.reference_selector.currentNode()

    def on_apply_dialog_folder_button(self):
        self.dir = self.dialog_folder_button.directory

    # def on_apply_train_folder_button(self):
    #     self.traindir = self.train_folder_button.directory

    def on_patch_button(self):
        if self.label_node and self.ref_node and self.dir:
            properties = {'useCompression': 0}
            volume_path = os.path.join(self.tmp, 'reference.nii')
            slicer.util.saveNode(self.ref_node, volume_path, properties)

            properties = {'useCompression': 0}
            label_path = os.path.join(self.tmp, 'labels.nii')
            slicer.util.saveNode(self.label_node, label_path, properties)

            if self.radio_imag.isChecked():
                legacy = True
            else:
                legacy = False

            get_patches(volume_path, label_path, self.dir, legacy)

            self.cleanup()

            # current_seeds_node = self.markups_selector.currentNode()
            # art_init = []
            # for n in range(current_seeds_node.GetNumberOfFiducials()):
            #     current = [0, 0, 0, 1]
            #     current_seeds_node.GetNthFiducialWorldCoordinates(n, current)
            #     transformRasToVolumeRas = vtk.vtkGeneralTransform()
            #     slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, self.ref_node.GetParentTransformNode(),
            #                                                          transformRasToVolumeRas)
            #     point_VolumeRas = transformRasToVolumeRas.TransformPoint(current[0:3])
            #     volumeRasToIjk = vtk.vtkMatrix4x4()
            #     self.ref_node.GetRASToIJKMatrix(volumeRasToIjk)
            #     point_Ijk = [0, 0, 0, 1]
            #     volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas, 1.0), point_Ijk)
            #     point_Ijk = [int(round(c)) for c in point_Ijk[0:3]]
            #
            #     art_init.append(point_Ijk)
            #
            # current_seeds_node = self.markups_selector_vein.currentNode()
            # vein_init = []
            # for n in range(current_seeds_node.GetNumberOfFiducials()):
            #     current = [0, 0, 0, 1]
            #     current_seeds_node.GetNthFiducialWorldCoordinates(n, current)
            #     transformRasToVolumeRas = vtk.vtkGeneralTransform()
            #     slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, self.ref_node.GetParentTransformNode(),
            #                                                          transformRasToVolumeRas)
            #     point_VolumeRas = transformRasToVolumeRas.TransformPoint(current[0:3])
            #     volumeRasToIjk = vtk.vtkMatrix4x4()
            #     self.ref_node.GetRASToIJKMatrix(volumeRasToIjk)
            #     point_Ijk = [0, 0, 0, 1]
            #     volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas, 1.0), point_Ijk)
            #     point_Ijk = [int(round(c)) for c in point_Ijk[0:3]]
            #
            #     vein_init.append(point_Ijk)
            #
            # ref_array = slicer.util.arrayFromVolume(self.ref_node)
            # ref_img = np.copy(ref_array / np.amax(ref_array))
            # for pos, i in enumerate(art_init):
            #     i = tuple(reversed(i))
            #     patch = (255 * ref_img[i[0] - 15:i[0] + 16, i[1] - 15:i[1] + 16, i[2] - 1:i[2] + 2]).astype(np.uint8)
            #     im = Image.fromarray(patch[::-1])
            #     path = os.path.join(self.dir, '{}/{}.png'.format(self.ref_node.GetName(), pos))
            #     im.save(path)
            #
            # label_array = slicer.util.arrayFromVolume(self.label_node)
            # label_img = np.copy(label_array)
            # for pos, i in enumerate(art_init):
            #     i = tuple(reversed(i))
            #     patch = (label_img[i[0] - 15:i[0] + 16, i[1] - 15:i[1] + 16, i[2]]).astype(np.uint8)
            #     im = Image.fromarray(patch[::-1])
            #     path = os.path.join(self.dir, '{}/{}.pgm'.format(self.ref_node.GetName(), pos))
            #     im.save(path)

    def on_clean_button(self):
        if self.dir:
            for f in os.listdir(self.dir):
                file_path = os.path.join(self.dir, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)

    def on_train_button(self):
        # if self.traindir:
        #     pass
        self.logic.train()

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
                                 tag='docker_vessels')
        mounts = {
            os.path.join(self.temp_folder, 'test_dataset', 'docker_infer'): {'bind': '/vessels_seg', 'mode': 'rw'}}
        self.client.containers.run(image='docker_vessels', auto_remove=True, volumes=mounts, name='DeepVessel')

        # cmd = 'docker ps -aq --no-trunc -f status=exited | xargs docker rm'
        # pipe(cmd, True, self.my_env)
        #
        # cmd = 'docker build -t test_docker_vessels . && docker run -v ' + \
        #       os.path.join(self.temp_folder, 'test_dataset',
        #                    'docker_infer') + ':/test_docker --name DeepVessel test_docker_vessels '
        # pipe(cmd, True, self.my_env)

        output_path = os.path.join(self.temp_folder, 'test_dataset', 'docker_infer', 'seg-label.nii')

        return output_path

    def train(self):
        self.client.containers.prune()
        self.client.images.build(path=self.temp_folder, dockerfile='training/Dockerfile',
                                 tag='docker_training')

        mounts = {
            os.path.join(self.temp_folder, 'training', 'files', 'log'): {'bind': '/log', 'mode': 'rw'}}
        self.client.containers.run(image='docker_training', auto_remove=True, volumes=mounts, name='TrainVessels')


class VesselSegmentationTest(unittest.TestCase):

    def __init__(self):
        super(VesselSegmentationTest, self).__init__()

    def __repr__(self):
        return 'VesselSegmentation(). Derived from {}'.format(unittest.TestCase)

    def __str__(self):
        return 'VesselSegmentation test class'

    def runTest(self, scenario=None):
        pass
