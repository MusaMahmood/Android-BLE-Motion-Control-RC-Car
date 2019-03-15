package com.yeolabgt.mahmoodms.wearablemotioncontrol

import android.app.Activity
import android.bluetooth.*
import android.content.Context
import android.content.Intent
import android.content.pm.ActivityInfo
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.ColorDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.support.v4.app.NavUtils
import android.support.v4.content.FileProvider
import android.support.v4.content.MimeTypeFilter
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import android.widget.ToggleButton

import com.androidplot.util.Redrawer
import com.google.common.primitives.Floats
import com.yeolabgt.mahmoodms.actblelibrary.ActBle
import kotlinx.android.synthetic.main.activity_device_control.*
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.io.File

import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

/**
 * Created by mahmoodms on 5/31/2016.
 * Android Activity for Controlling Bluetooth LE Device Connectivity
 */

class DeviceControlActivity : Activity(), ActBle.ActBleListener, TensorflowOptionsMenu.NoticeDialogListener {
    // Graphing Variables:
    private var mGraphInitializedBoolean = false
    private var mGraphAdapterMotionAX: GraphAdapter? = null
    private var mGraphAdapterMotionAY: GraphAdapter? = null
    private var mGraphAdapterMotionAZ: GraphAdapter? = null
    private var mGraphAdapterMotionGX: GraphAdapter? = null
    private var mGraphAdapterMotionGY: GraphAdapter? = null
    private var mGraphAdapterMotionGZ: GraphAdapter? = null
    private var mGraphAdapterMotionMX: GraphAdapter? = null
    private var mGraphAdapterMotionMY: GraphAdapter? = null
    private var mGraphAdapterMotionMZ: GraphAdapter? = null
    private var mMotionDataPlotAdapter: XYPlotAdapter? = null
    private var mMotionDataPlotAdapter2: XYPlotAdapter? = null
    private var mMotionDataPlotAdapter3: XYPlotAdapter? = null
    //Device Information
    private var mBleInitializedBoolean = false
    private lateinit var mBluetoothGattArray: Array<BluetoothGatt?>
    private var mLedWheelchairControlService: BluetoothGattService? = null
    private var mWheelchairGattIndex: Int = 0
    //Classification
    private var mWheelchairControl = false //Default classifier.
    private var mActBle: ActBle? = null
    private var mDeviceName: String? = null
    private var mDeviceAddress: String? = null
    private var mConnected: Boolean = false
    private var mMSBFirst = false
    //Connecting to Multiple Devices
    private var deviceMacAddresses: Array<String>? = null
    //UI Elements - TextViews, Buttons, etc
    private var mBatteryLevel: TextView? = null
    private var mDataRate: TextView? = null
    private var mChannelSelect: ToggleButton? = null
    private var menu: Menu? = null
    //Data throughput counter
    private var mLastTime: Long = 0
    private var points = 0
    private val mTimerHandler = Handler()
    private var mTimerEnabled = false
    //Data Variables:
    private val batteryWarning = 20//
    private var dataRate: Double = 0.toDouble()
    //Tensorflow:
    private var mTFRunModel = false
    private var mClassifierInput = DoubleArray(20*6)
    private var mTFInferenceInterface: TensorFlowInferenceInterface? = null
    private var mOutputScoresNames: Array<String>? = null
    private var mTensorflowSolutionIndex = 0
    private var mTensorflowWindowSize = 20
    private var mTensorflowXDim = 6 // x-dimension
    private var mTensorflowYDim = 20 // y-dimension
    private var mNumberOfClassifierCalls = 0
    private val mTimeStamp: String
        get() = SimpleDateFormat("yyyy.MM.dd_HH.mm.ss", Locale.US).format(Date())

    private val mClassifyThread = Runnable {
        if (mTFRunModel) {
            mOutputScoresNames = arrayOf(OUTPUT_DATA_FEED)
            val outputScores = FloatArray(5)
            val mTensorflowInputFeed: FloatArray = Floats.toArray(mClassifierInput.asList())
            Log.i(TAG, "mTensorflowInputFeed: " + Arrays.toString(mTensorflowInputFeed))
            // Feed Data:
            mTFInferenceInterface!!.feed("keep_prob", floatArrayOf(1f))
            mTFInferenceInterface!!.feed(INPUT_DATA_FEED, mTensorflowInputFeed, mTensorflowXDim.toLong(), mTensorflowYDim.toLong())
            mTFInferenceInterface!!.run(mOutputScoresNames)
            mTFInferenceInterface!!.fetch(OUTPUT_DATA_FEED, outputScores)
            Log.i(TAG, "outputScores: "+Arrays.toString(outputScores))
            val yTF = DataChannel.getIndexOfLargest(outputScores)
            Log.i(TAG, "CALL#" + mNumberOfClassifierCalls.toString() + ":\n" +
                    "TF outputScores: " + Arrays.toString(outputScores))
            val s = "[" + yTF.toString() + "]"
            runOnUiThread { myfit!!.text = s }
            mNumberOfClassifierCalls++
            executeWheelchairCommand(yTF)
        } else {
            val s = ""
            runOnUiThread { myfit!!.text = s }
        }
    }

    private fun showNoticeDialog() {
        val dialog = TensorflowOptionsMenu()
        dialog.show(fragmentManager, "TFOM")
    }

    @Override
    override fun onTensorflowOptionsClick(integerValue: Int) {
        enableTensorFlowModel(File(MODEL_FILENAME), integerValue)
    }

    private fun enableTensorFlowModel(embeddedModel: File, integerValue: Int) {
        mTensorflowSolutionIndex = integerValue
        val customModelPath = Environment.getExternalStorageDirectory().absolutePath + "/Download/tensorflow_assets/"
        // Hard-coded Strings of Model Names
        // NOTE: Zero index is an empty string (no model)
        val customModel = arrayOf("", "motion_ctrl_opt_CNN-1-a.parametricrelu.[0.1]-drop0.5-fc.64.relu-lr.1e-3-k.[5]")

        val tensorflowModelLocation = customModelPath + customModel[integerValue] + ".pb"
        for (s in customModel) {
            val tempPath = customModelPath + s + ".pb"
            Log.e(TAG, "Model " + tempPath + " exists? " + File(tempPath).exists().toString())
        }
        mTensorflowWindowSize = 20
        mTensorflowXDim = 6
        mTensorflowYDim = 20
        Log.e(TAG, "Input Length: 6x" + mTensorflowWindowSize + " Output = " + mTensorflowXDim + "x" + mTensorflowYDim)
        when {
            File(tensorflowModelLocation).exists() -> {
                mTFInferenceInterface = TensorFlowInferenceInterface(assets, tensorflowModelLocation)
                //Reset counter:
                mNumberOfClassifierCalls = 1
                mTFRunModel = true
                Log.i(TAG, "Tensorflow: customModel loaded")
            }
            embeddedModel.exists() -> { //Check if there's a model included:
                mChannelSelect!!.isChecked = false // tensorflowClassificationSwitch
                mTFRunModel = false
                Toast.makeText(applicationContext, "No TF Model Found!", Toast.LENGTH_LONG).show()
            }
            else -> { // No model found, continuing with original (reset switch)
                mChannelSelect!!.isChecked = false
                mTFRunModel = false
                Toast.makeText(applicationContext, "No TF Model Found!", Toast.LENGTH_LONG).show()
            }
        }
        if (mTFRunModel) {
            Toast.makeText(applicationContext, "TF Model Loaded", Toast.LENGTH_SHORT).show()
        }
    }

    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_device_control)
        //Set orientation of device based on screen type/size:
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        //Receive Intents:
        val intent = intent
        deviceMacAddresses = intent.getStringArrayExtra(MainActivity.INTENT_DEVICES_KEY)
        val deviceDisplayNames = intent.getStringArrayExtra(MainActivity.INTENT_DEVICES_NAMES)
        mDeviceName = deviceDisplayNames[0]
        mDeviceAddress = deviceMacAddresses!![0]
        Log.d(TAG, "Device Names: " + Arrays.toString(deviceDisplayNames))
        Log.d(TAG, "Device MAC Addresses: " + Arrays.toString(deviceMacAddresses))
        Log.d(TAG, Arrays.toString(deviceMacAddresses))
        //Set up action bar:
        if (actionBar != null) {
            actionBar!!.setDisplayHomeAsUpEnabled(true)
        }
        val actionBar = actionBar
        actionBar!!.setBackgroundDrawable(ColorDrawable(Color.parseColor("#6078ef")))
        //Flag to keep screen on (stay-awake):
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        //Set up TextViews
        val mExportButton = findViewById<Button>(R.id.button_export)
        mBatteryLevel = findViewById(R.id.batteryText)
        mDataRate = findViewById(R.id.dataRate)
        mDataRate!!.text = "..."
        val ab = getActionBar()
        ab!!.title = mDeviceName
        ab.subtitle = mDeviceAddress
        //Initialize Bluetooth
        if (!mBleInitializedBoolean) initializeBluetoothArray()
        mLastTime = System.currentTimeMillis()
        //UI Listeners
        mChannelSelect = findViewById(R.id.toggleButtonGraph)
        buttonS.visibility = View.INVISIBLE
        buttonF.visibility = View.INVISIBLE
        buttonL.visibility = View.INVISIBLE
        buttonR.visibility = View.INVISIBLE
        buttonReverse.visibility = View.INVISIBLE
        mChannelSelect!!.setOnCheckedChangeListener { _, b ->
            mWheelchairControl = b
            val viewVisibility = if (b) View.VISIBLE else View.INVISIBLE
            buttonS.visibility = viewVisibility
            buttonF.visibility = viewVisibility
            buttonL.visibility = viewVisibility
            buttonR.visibility = viewVisibility
            buttonReverse.visibility = viewVisibility
            if (b) {
                showNoticeDialog()
            } else {
                //Reset counter:
                mTFRunModel = false
                mNumberOfClassifierCalls = 1
            }
        }
        batteryText.visibility = View.GONE
        mExportButton.setOnClickListener { exportData() }
        buttonS.setOnClickListener { executeWheelchairCommand(0) }
        buttonF.setOnClickListener { executeWheelchairCommand(1) }
        buttonL.setOnClickListener { executeWheelchairCommand(2) }
        buttonR.setOnClickListener { executeWheelchairCommand(3) }
        buttonReverse.setOnClickListener { executeWheelchairCommand(4) }
    }

    private fun exportData() {
        try {
            terminateDataFileWriter()
        } catch (e: IOException) {
            Log.e(TAG, "IOException in saveDataFile")
            e.printStackTrace()
        }
        val files = ArrayList<Uri>()
        val context = applicationContext
        if(mSaveFileMPU!=null) {
            val uii2 = FileProvider.getUriForFile(context, context.packageName + ".provider", mSaveFileMPU!!.file)
            files.add(uii2)
        }
        val exportData = Intent(Intent.ACTION_SEND_MULTIPLE)
        exportData.putExtra(Intent.EXTRA_SUBJECT, "ECG Sensor Data Export Details")
        exportData.putParcelableArrayListExtra(Intent.EXTRA_STREAM, files)
        exportData.type = "text/html"
        startActivity(exportData)
    }

    @Throws(IOException::class)
    private fun terminateDataFileWriter() {
        mSaveFileMPU?.terminateDataFileWriter()
    }

    public override fun onResume() {
        jmainInitialization(true)
        if (mRedrawer != null) {
            mRedrawer!!.start()
        }
        super.onResume()
    }

    override fun onPause() {
        if (mRedrawer != null) mRedrawer!!.pause()
        super.onPause()
    }

    private fun initializeBluetoothArray() {
        val mBluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        val mBluetoothDeviceArray = arrayOfNulls<BluetoothDevice>(deviceMacAddresses!!.size)
        Log.d(TAG, "Device Addresses: " + Arrays.toString(deviceMacAddresses))
        if (deviceMacAddresses != null) {
            for (i in deviceMacAddresses!!.indices) {
                mBluetoothDeviceArray[i] = mBluetoothManager.adapter.getRemoteDevice(deviceMacAddresses!![i])
            }
        } else {
            Log.e(TAG, "No Devices Queued, Restart!")
            Toast.makeText(this, "No Devices Queued, Restart!", Toast.LENGTH_SHORT).show()
        }
        mActBle = ActBle(this, mBluetoothManager, this)
        mBluetoothGattArray = Array(deviceMacAddresses!!.size, { i -> mActBle!!.connect(mBluetoothDeviceArray[i]) })
        for (i in mBluetoothDeviceArray.indices) {
            Log.e(TAG, "Connecting to Device: " + (mBluetoothDeviceArray[i]!!.name + " " + mBluetoothDeviceArray[i]!!.address))
            if ("WheelchairControl" == mBluetoothDeviceArray[i]!!.name) {
                mWheelchairGattIndex = i
                Log.e(TAG, "mWheelchairGattIndex: " + mWheelchairGattIndex)
                continue //we are done initializing
            }

            val str = mBluetoothDeviceArray[i]!!.name.toLowerCase()
            when {
                str.contains("8k") -> {
                    mSampleRate = 8000
                }
                str.contains("4k") -> {
                    mSampleRate = 4000
                }
                str.contains("2k") -> {
                    mSampleRate = 2000
                }
                str.contains("1k") -> {
                    mSampleRate = 1000
                }
                str.contains("500") -> {
                    mSampleRate = 500
                }
                else -> {
                    mSampleRate = 250
                }
            }
            mPacketBuffer = mSampleRate / 250
            Log.e(TAG, "mSampleRate: " + mSampleRate + "Hz")
            if (!mGraphInitializedBoolean) setupGraph()
        }
        mBleInitializedBoolean = true
    }

    private fun createNewFileMPU() {
        val directory = "/MPUData"
        val fileNameTimeStamped = "MPUData_" + mTimeStamp
        if (mSaveFileMPU == null) {
            Log.e(TAG, "fileTimeStamp: " + fileNameTimeStamped)
            mSaveFileMPU = SaveDataFile(directory, fileNameTimeStamped,
                    16, 0.032, true, false)
        } else if (!mSaveFileMPU!!.initialized) {
            Log.e(TAG, "New Filename: " + fileNameTimeStamped)
            mSaveFileMPU?.createNewFile(directory, fileNameTimeStamped)
        }
    }

    private fun executeWheelchairCommand(command: Int) {
        val bytes = ByteArray(1)
        when (command) {
            0 -> bytes[0] = 0x00.toByte()
            1 -> bytes[0] = 0x01.toByte() // Stop
            2 -> bytes[0] = 0xF0.toByte() // Rotate Left
            3 -> bytes[0] = 0x0F.toByte() // Rotate Right ??
            4 -> bytes[0] = 0xFF.toByte() // TODO: 6/27/2017 Disconnect instead of reverse?
            else -> {
            }
        }
        if (mLedWheelchairControlService != null && mWheelchairControl) {
            Log.e(TAG, "SendingCommand: " + command.toString())
            Log.e(TAG, "SendingCommand (byte): " + DataChannel.byteArrayToHexString(bytes))
            mActBle!!.writeCharacteristic(mBluetoothGattArray[mWheelchairGattIndex]!!, mLedWheelchairControlService!!.getCharacteristic(AppConstant.CHAR_WHEELCHAIR_CONTROL), bytes)
        }
    }

    private fun setupGraph() {
        // Initialize our XYPlot reference:
        mGraphAdapterMotionGX = GraphAdapter(375, "Gyr X", false, rgbToInt(209, 69, 69))
        mGraphAdapterMotionGY = GraphAdapter(375, "Gyr Y", false, rgbToInt(0xFF, 0xB3, 0x00))
        mGraphAdapterMotionGZ = GraphAdapter(375, "Gyr Z", false, rgbToInt(0x66, 0xB2, 0xFF))
        mGraphAdapterMotionAX = GraphAdapter(375, "Acc X", false, Color.RED)
        mGraphAdapterMotionAY = GraphAdapter(375, "Acc Y", false, Color.GREEN)
        mGraphAdapterMotionAZ = GraphAdapter(375, "Acc Z", false, Color.BLUE)
        mGraphAdapterMotionMX = GraphAdapter(375, "Magn X", false, rgbToInt(66, 244, 217))
        mGraphAdapterMotionMY = GraphAdapter(375, "Magn Y", false, rgbToInt(226, 92, 9))
        mGraphAdapterMotionMZ = GraphAdapter(375, "Magn Z", false, rgbToInt(182, 8, 226))
        //PLOT CH1 By default
        mMotionDataPlotAdapter = XYPlotAdapter(findViewById(R.id.motionDataPlot), "Time (s)", "Acc (g)", 375.0)
        mMotionDataPlotAdapter?.xyPlot!!.addSeries(mGraphAdapterMotionAX?.series, mGraphAdapterMotionAX?.lineAndPointFormatter)
        mMotionDataPlotAdapter?.xyPlot!!.addSeries(mGraphAdapterMotionAY?.series, mGraphAdapterMotionAY?.lineAndPointFormatter)
        mMotionDataPlotAdapter?.xyPlot!!.addSeries(mGraphAdapterMotionAZ?.series, mGraphAdapterMotionAZ?.lineAndPointFormatter)
        mMotionDataPlotAdapter2 = XYPlotAdapter(findViewById(R.id.motionDataPlot2), "Time (s)", "Gyro (r/s)", 375.0)
        mMotionDataPlotAdapter2?.xyPlot!!.addSeries(mGraphAdapterMotionGX?.series, mGraphAdapterMotionGX?.lineAndPointFormatter)
        mMotionDataPlotAdapter2?.xyPlot!!.addSeries(mGraphAdapterMotionGY?.series, mGraphAdapterMotionGY?.lineAndPointFormatter)
        mMotionDataPlotAdapter2?.xyPlot!!.addSeries(mGraphAdapterMotionGZ?.series, mGraphAdapterMotionGZ?.lineAndPointFormatter)
        mMotionDataPlotAdapter3 = XYPlotAdapter(findViewById(R.id.motionDataPlot3), "Time (s)", "Magn (uT)", 375.0)
        mMotionDataPlotAdapter3?.xyPlot!!.addSeries(mGraphAdapterMotionMX?.series, mGraphAdapterMotionMX?.lineAndPointFormatter)
        mMotionDataPlotAdapter3?.xyPlot!!.addSeries(mGraphAdapterMotionMY?.series, mGraphAdapterMotionMY?.lineAndPointFormatter)
        mMotionDataPlotAdapter3?.xyPlot!!.addSeries(mGraphAdapterMotionMZ?.series, mGraphAdapterMotionMZ?.lineAndPointFormatter)
        val xyPlotList = listOf(mMotionDataPlotAdapter?.xyPlot, mMotionDataPlotAdapter2?.xyPlot, mMotionDataPlotAdapter3?.xyPlot)
        mRedrawer = Redrawer(xyPlotList, 30f, false)
        mRedrawer!!.start()
        mGraphInitializedBoolean = true
        mGraphAdapterMotionAX?.setxAxisIncrement(0.032)
        mGraphAdapterMotionAX?.setSeriesHistoryDataPoints(375)
        mGraphAdapterMotionAY?.setxAxisIncrement(0.032)
        mGraphAdapterMotionAY?.setSeriesHistoryDataPoints(375)
        mGraphAdapterMotionAZ?.setxAxisIncrement(0.032)
        mGraphAdapterMotionAZ?.setSeriesHistoryDataPoints(375)
        mGraphAdapterMotionGX?.setxAxisIncrement(0.032)
        mGraphAdapterMotionGX?.setSeriesHistoryDataPoints(375)
        mGraphAdapterMotionGY?.setxAxisIncrement(0.032)
        mGraphAdapterMotionGY?.setSeriesHistoryDataPoints(375)
        mGraphAdapterMotionGZ?.setxAxisIncrement(0.032)
        mGraphAdapterMotionGZ?.setSeriesHistoryDataPoints(375)
        mGraphAdapterMotionMX?.setxAxisIncrement(0.032)
        mGraphAdapterMotionMX?.setSeriesHistoryDataPoints(375)
        mGraphAdapterMotionMY?.setxAxisIncrement(0.032)
        mGraphAdapterMotionMY?.setSeriesHistoryDataPoints(375)
        mGraphAdapterMotionMZ?.setxAxisIncrement(0.032)
        mGraphAdapterMotionMZ?.setSeriesHistoryDataPoints(375)
    }

    private fun rgbToInt(R: Int, G: Int, B: Int):Int {
        val r = R shl 16 and 0x00FF0000
        val g = G shl 8 and 0x0000FF00
        val b = B and 0x000000FF

        return -0x1000000 or r or g or b
    }

    private fun setNameAddress(name_action: String?, address_action: String?) {
        val name = menu!!.findItem(R.id.action_title)
        val address = menu!!.findItem(R.id.action_address)
        name.title = name_action
        address.title = address_action
        invalidateOptionsMenu()
    }

    override fun onDestroy() {
        mRedrawer?.finish()
        disconnectAllBLE()
        try {
            terminateDataFileWriter()
        } catch (e: IOException) {
            Log.e(TAG, "IOException in saveDataFile")
            e.printStackTrace()
        }

        stopMonitoringRssiValue()
        jmainInitialization(false) //Just a technicality, doesn't actually do anything
        super.onDestroy()
    }

    private fun disconnectAllBLE() {
        if (mActBle != null) {
            for (bluetoothGatt in mBluetoothGattArray) {
                mActBle!!.disconnect(bluetoothGatt!!)
                mConnected = false
                resetMenuBar()
            }
        }
    }

    private fun resetMenuBar() {
        runOnUiThread {
            if (menu != null) {
                menu!!.findItem(R.id.menu_connect).isVisible = true
                menu!!.findItem(R.id.menu_disconnect).isVisible = false
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_device_control, menu)
        menuInflater.inflate(R.menu.actionbar_item, menu)
        if (mConnected) {
            menu.findItem(R.id.menu_connect).isVisible = false
            menu.findItem(R.id.menu_disconnect).isVisible = true
        } else {
            menu.findItem(R.id.menu_connect).isVisible = true
            menu.findItem(R.id.menu_disconnect).isVisible = false
        }
        this.menu = menu
        setNameAddress(mDeviceName, mDeviceAddress)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.menu_connect -> {
                if (mActBle != null) {
                    initializeBluetoothArray()
                }
                connect()
                return true
            }
            R.id.menu_disconnect -> {
                if (mActBle != null) {
                    disconnectAllBLE()
                }
                return true
            }
            android.R.id.home -> {
                if (mActBle != null) {
                    disconnectAllBLE()
                }
                NavUtils.navigateUpFromSameTask(this)
                onBackPressed()
                return true
            }
            R.id.action_settings -> {
                launchSettingsMenu()
                return true
            }
            R.id.action_export -> {
                exportData()
                return true
            }
        }
        return super.onOptionsItemSelected(item)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == 1) {
            val context = applicationContext
            //UI Stuff:
            val chSel = PreferencesFragment.channelSelect(context)
            //File Save Stuff
            mChannelSelect!!.isChecked = chSel
        }
        super.onActivityResult(requestCode, resultCode, data)
    }

    private fun launchSettingsMenu() {
        val intent = Intent(applicationContext, SettingsActivity::class.java)
        startActivityForResult(intent, 1)
    }

    private fun connect() {
        runOnUiThread {
            val menuItem = menu!!.findItem(R.id.action_status)
            menuItem.title = "Connecting..."
        }
    }

    override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
        Log.i(TAG, "onServicesDiscovered")
        if (status == BluetoothGatt.GATT_SUCCESS) {
            for (service in gatt.services) {
                if (service == null || service.uuid == null) {
                    continue
                }
                if (AppConstant.SERVICE_DEVICE_INFO == service.uuid) {
                    //Read the device serial number (if available)
                    if (service.getCharacteristic(AppConstant.CHAR_SERIAL_NUMBER) != null) {
                        mActBle!!.readCharacteristic(gatt, service.getCharacteristic(AppConstant.CHAR_SERIAL_NUMBER))
                    }
                    //Read the device software version (if available)
                    if (service.getCharacteristic(AppConstant.CHAR_SOFTWARE_REV) != null) {
                        mActBle!!.readCharacteristic(gatt, service.getCharacteristic(AppConstant.CHAR_SOFTWARE_REV))
                    }
                }

                if (AppConstant.SERVICE_WHEELCHAIR_CONTROL == service.uuid) {
                    mLedWheelchairControlService = service
                    Log.i(TAG, "BLE Wheelchair Control Service found")
                }

                if (AppConstant.SERVICE_EEG_SIGNAL == service.uuid) {
                    if (service.getCharacteristic(AppConstant.CHAR_EEG_CH1_SIGNAL) != null) {
                        mActBle!!.setCharacteristicNotifications(gatt, service.getCharacteristic(AppConstant.CHAR_EEG_CH1_SIGNAL), true)
                    }
                    if (service.getCharacteristic(AppConstant.CHAR_EEG_CH2_SIGNAL) != null) {
                        mActBle!!.setCharacteristicNotifications(gatt, service.getCharacteristic(AppConstant.CHAR_EEG_CH2_SIGNAL), true)
                    }
                    if (service.getCharacteristic(AppConstant.CHAR_EEG_CH3_SIGNAL) != null) {
                        mActBle!!.setCharacteristicNotifications(gatt, service.getCharacteristic(AppConstant.CHAR_EEG_CH3_SIGNAL), true)
                    }
                    if (service.getCharacteristic(AppConstant.CHAR_EEG_CH4_SIGNAL) != null) {
                        mActBle!!.setCharacteristicNotifications(gatt, service.getCharacteristic(AppConstant.CHAR_EEG_CH4_SIGNAL), true)
                    }
                }

                if (AppConstant.SERVICE_BATTERY_LEVEL == service.uuid) { //Read the device battery percentage
                    mActBle!!.readCharacteristic(gatt, service.getCharacteristic(AppConstant.CHAR_BATTERY_LEVEL))
                    mActBle!!.setCharacteristicNotifications(gatt, service.getCharacteristic(AppConstant.CHAR_BATTERY_LEVEL), true)
                }

                if (AppConstant.SERVICE_MPU == service.uuid) {
                    mActBle!!.setCharacteristicNotifications(gatt, service.getCharacteristic(AppConstant.CHAR_MPU_COMBINED), true)
                    //TODO: INITIALIZE MPU FILE HERE:
                    mMPU = DataChannel(false, true, 0)
//                    mSaveFileMPU = null
                    createNewFileMPU()
                }
            }
            //Run process only once:
            mActBle?.runProcess()
        }
    }

    override fun onCharacteristicRead(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic, status: Int) {
        Log.i(TAG, "onCharacteristicRead")
        if (status == BluetoothGatt.GATT_SUCCESS) {
            if (AppConstant.CHAR_BATTERY_LEVEL == characteristic.uuid) {
                if (characteristic.value != null) {
                    val batteryLevel = characteristic.getIntValue(BluetoothGattCharacteristic.FORMAT_UINT16, 0)
                    updateBatteryStatus(batteryLevel)
                    Log.i(TAG, "Battery Level :: " + batteryLevel)
                }
            }
        } else {
            Log.e(TAG, "onCharacteristic Read Error" + status)
        }
    }

    override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
        if (AppConstant.CHAR_BATTERY_LEVEL == characteristic.uuid) {
            val batteryLevel = characteristic.getIntValue(BluetoothGattCharacteristic.FORMAT_UINT16, 0)!!
            updateBatteryStatus(batteryLevel)
        }

        if (AppConstant.CHAR_MPU_COMBINED == characteristic.uuid) {
            val dataMPU = characteristic.value
            getDataRateBytes(dataMPU.size)
            mMPU!!.handleNewData(dataMPU)
            addToGraphBufferMPU(mMPU!!)
            // Get data from buffer!: type: Concatenataed DoubleArray (sizeof 6*20).
            mSaveFileMPU!!.exportDataWithTimestampMPU(mMPU!!.characteristicDataPacketBytes)
//            val classifyTaskThread = Thread(mClassifyThread)
//            classifyTaskThread.start()
            if (mSaveFileMPU!!.mLinesWrittenCurrentFile > 1048576) {
                mSaveFileMPU!!.terminateDataFileWriter()
                createNewFileMPU()
            }
            //TODO: Classify!
        }
    }

    private fun addToGraphBufferMPU(dataChannel: DataChannel) {
        if (dataChannel.dataBuffer!=null) {
            for (i in 0 until dataChannel.dataBuffer!!.size / 18) {
                mGraphAdapterMotionAX?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUAccel(dataChannel.dataBuffer!![18 * i], dataChannel.dataBuffer!![18 * i + 1]), mTimestampIdxMPU)
                mGraphAdapterMotionAY?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUAccel(dataChannel.dataBuffer!![18 * i + 2], dataChannel.dataBuffer!![18 * i + 3]), mTimestampIdxMPU)
                mGraphAdapterMotionAZ?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUAccel(dataChannel.dataBuffer!![18 * i + 4], dataChannel.dataBuffer!![18 * i + 5]), mTimestampIdxMPU)
                mGraphAdapterMotionGX?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUGyro(dataChannel.dataBuffer!![18 * i + 6], dataChannel.dataBuffer!![18 * i + 7]), mTimestampIdxMPU)
                mGraphAdapterMotionGY?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUGyro(dataChannel.dataBuffer!![18 * i + 8], dataChannel.dataBuffer!![18 * i + 9]), mTimestampIdxMPU)
                mGraphAdapterMotionGZ?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUGyro(dataChannel.dataBuffer!![18 * i + 10], dataChannel.dataBuffer!![18 * i + 11]), mTimestampIdxMPU)
                mGraphAdapterMotionMX?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUGyro(dataChannel.dataBuffer!![18 * i + 12], dataChannel.dataBuffer!![18 * i + 13]), mTimestampIdxMPU)
                mGraphAdapterMotionMY?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUGyro(dataChannel.dataBuffer!![18 * i + 14], dataChannel.dataBuffer!![18 * i + 15]), mTimestampIdxMPU)
                mGraphAdapterMotionMZ?.addDataPointTimeDomain(DataChannel.bytesToDoubleMPUGyro(dataChannel.dataBuffer!![18 * i + 16], dataChannel.dataBuffer!![18 * i + 17]), mTimestampIdxMPU)
                mTimestampIdxMPU += 1
            }
        }
        dataChannel.resetBuffer()
    }

    private fun getDataRateBytes(bytes: Int) {
        val mCurrentTime = System.currentTimeMillis()
        points += bytes
        if (mCurrentTime > mLastTime + 5000) {
            dataRate = (points / 5).toDouble()
            points = 0
            mLastTime = mCurrentTime
            Log.e(" DataRate:", dataRate.toString() + " Bytes/s")
            runOnUiThread {
                val s = dataRate.toString() + " Bytes/s"
                mDataRate!!.text = s
            }
        }
    }

    override fun onReadRemoteRssi(gatt: BluetoothGatt, rssi: Int, status: Int) {
        uiRssiUpdate(rssi)
    }

    override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
        when (newState) {
            BluetoothProfile.STATE_CONNECTED -> {
                mConnected = true
                runOnUiThread {
                    if (menu != null) {
                        menu!!.findItem(R.id.menu_connect).isVisible = false
                        menu!!.findItem(R.id.menu_disconnect).isVisible = true
                    }
                }
                Log.i(TAG, "Connected")
                updateConnectionState(getString(R.string.connected))
                invalidateOptionsMenu()
                runOnUiThread {
                    mDataRate!!.setTextColor(Color.BLACK)
                    mDataRate!!.setTypeface(null, Typeface.NORMAL)
                }
                //Start the service discovery:
                gatt.discoverServices()
                startMonitoringRssiValue()
            }
            BluetoothProfile.STATE_CONNECTING -> {
            }
            BluetoothProfile.STATE_DISCONNECTING -> {
            }
            BluetoothProfile.STATE_DISCONNECTED -> {
                mConnected = false
                runOnUiThread {
                    if (menu != null) {
                        menu!!.findItem(R.id.menu_connect).isVisible = true
                        menu!!.findItem(R.id.menu_disconnect).isVisible = false
                    }
                }
                Log.i(TAG, "Disconnected")
                runOnUiThread {
                    mDataRate!!.setTextColor(Color.RED)
                    mDataRate!!.setTypeface(null, Typeface.BOLD)
                    mDataRate!!.text = HZ
                }
                updateConnectionState(getString(R.string.disconnected))
                stopMonitoringRssiValue()
                invalidateOptionsMenu()
            }
            else -> {
            }
        }
    }

    private fun startMonitoringRssiValue() {
        readPeriodicallyRssiValue(true)
    }

    private fun stopMonitoringRssiValue() {
        readPeriodicallyRssiValue(false)
    }

    private fun readPeriodicallyRssiValue(repeat: Boolean) {
        mTimerEnabled = repeat
        // check if we should stop checking RSSI value
        if (!mConnected || !mTimerEnabled) {
            mTimerEnabled = false
            return
        }

        mTimerHandler.postDelayed(Runnable {
            if (!mConnected) {
                mTimerEnabled = false
                return@Runnable
            }
            // request RSSI value
            mBluetoothGattArray[0]!!.readRemoteRssi()
            // add call it once more in the future
            readPeriodicallyRssiValue(mTimerEnabled)
        }, RSSI_UPDATE_TIME_INTERVAL.toLong())
    }

    override fun onCharacteristicWrite(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic, status: Int) {
        Log.i(TAG, "onCharacteristicWrite :: Status:: " + status)
    }

    override fun onDescriptorWrite(gatt: BluetoothGatt, descriptor: BluetoothGattDescriptor, status: Int) {}

    override fun onDescriptorRead(gatt: BluetoothGatt, descriptor: BluetoothGattDescriptor, status: Int) {
        Log.i(TAG, "onDescriptorRead :: Status:: " + status)
    }

    override fun onError(errorMessage: String) {
        Log.e(TAG, "Error:: " + errorMessage)
    }

    private fun updateConnectionState(status: String) {
        runOnUiThread {
            if (status == getString(R.string.connected)) {
                Toast.makeText(applicationContext, "Device Connected!", Toast.LENGTH_SHORT).show()
            } else if (status == getString(R.string.disconnected)) {
                Toast.makeText(applicationContext, "Device Disconnected!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun updateBatteryStatus(integerValue: Int) {
        val status: String
        val convertedBatteryVoltage = integerValue.toDouble() / 4096.0 * 7.20
        //Because TPS63001 dies below 1.8V, we need to set up a linear fit between 1.8-4.2V
        //Anything over 4.2V = 100%
        val finalPercent: Double = when {
            125.0 / 3.0 * convertedBatteryVoltage - 75.0 > 100.0 -> 100.0
            125.0 / 3.0 * convertedBatteryVoltage - 75.0 < 0.0 -> 0.0
            else -> 125.0 / 3.0 * convertedBatteryVoltage - 75.0
        }
        Log.e(TAG, "Battery Integer Value: " + integerValue.toString())
        Log.e(TAG, "ConvertedBatteryVoltage: " + String.format(Locale.US, "%.5f", convertedBatteryVoltage) + "V : " + String.format(Locale.US, "%.3f", finalPercent) + "%")
        status = String.format(Locale.US, "%.1f", finalPercent) + "%"
        runOnUiThread {
            if (finalPercent <= batteryWarning) {
                mBatteryLevel!!.setTextColor(Color.RED)
                mBatteryLevel!!.setTypeface(null, Typeface.BOLD)
                Toast.makeText(applicationContext, "Charge Battery, Battery Low " + status, Toast.LENGTH_SHORT).show()
            } else {
                mBatteryLevel!!.setTextColor(Color.GREEN)
                mBatteryLevel!!.setTypeface(null, Typeface.BOLD)
            }
            mBatteryLevel!!.text = status
        }
    }

    private fun uiRssiUpdate(rssi: Int) {
        runOnUiThread {
            val menuItem = menu!!.findItem(R.id.action_rssi)
            val statusActionItem = menu!!.findItem(R.id.action_status)
            val valueOfRSSI = rssi.toString() + " dB"
            menuItem.title = valueOfRSSI
            if (mConnected) {
                val newStatus = "Status: " + getString(R.string.connected)
                statusActionItem.title = newStatus
            } else {
                val newStatus = "Status: " + getString(R.string.disconnected)
                statusActionItem.title = newStatus
            }
        }
    }

    private external fun jmainInitialization(initialize: Boolean): Int

    companion object {
        val HZ = "0 Hz"
        private val TAG = DeviceControlActivity::class.java.simpleName
        var mRedrawer: Redrawer? = null
        // Power Spectrum Graph Data:
        private var mSampleRate = 250
        //Data Channel Classes
        internal var mCh1: DataChannel? = null
        internal var mCh2: DataChannel? = null
        internal var mMPU: DataChannel? = null
        private var mPacketBuffer = 6
        private var mTimestampIdxMPU = 0
        //RSSI:
        const val RSSI_UPDATE_TIME_INTERVAL = 2000
        var mSSVEPClass = 0.0
        const val INPUT_DATA_FEED = "input"
        const val OUTPUT_DATA_FEED = "output"
        //Save Data File
//        private var mPrimarySaveDataFile: SaveDataFile? = null
        private var mSaveFileMPU: SaveDataFile? = null
        private const val MODEL_FILENAME = "file:///android_asset/opt_ssvep_net.pb"
        init {
            System.loadLibrary("ecg-lib")
        }
    }
}
