/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:11:47 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Function;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.AbstractCommandLineRunner;
import com.google.javascript.jscomp.CommandLineRunner;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSModuleGraph;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.Result;
import com.google.javascript.jscomp.SourceMap;
import com.google.protobuf.ByteString;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;
import java.util.Stack;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.System;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AbstractCommandLineRunner_ESTest extends AbstractCommandLineRunner_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      assertTrue(commandLineRunner0.shouldRunCompiler());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String[] stringArray0 = new String[9];
      stringArray0[0] = "[";
      stringArray0[1] = "ST^QFs|uWKu";
      stringArray0[2] = "";
      stringArray0[3] = "9";
      stringArray0[4] = "Normalizing";
      stringArray0[5] = "2`t";
      stringArray0[6] = "=piU|";
      stringArray0[7] = "";
      stringArray0[8] = "INHERITED";
      ByteString.Output byteString_Output0 = ByteString.newOutput();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteString_Output0, true);
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0, mockPrintStream0, mockPrintStream0);
      commandLineRunner0.initOptionsFromFlags((CompilerOptions) null);
      assertFalse(commandLineRunner0.shouldRunCompiler());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "com.google.javascript.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Compiler compiler0 = commandLineRunner0.getCompiler();
      assertNull(compiler0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "com.google.javascript.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      // Undeclared exception!
      try { 
        commandLineRunner0.run();
        fail("Expecting exception: System.SystemExitException");
      
      } catch(System.SystemExitException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "com.google.javascript.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      List<String> list0 = ResourceBundle.Control.FORMAT_DEFAULT;
      try { 
        commandLineRunner0.createJsModules(list0, list0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Expected 2-4 colon-delimited parts in module spec: java.class
         //
         verifyException("com.google.javascript.jscomp.AbstractCommandLineRunner", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      CompilerOptions.TweakProcessing compilerOptions_TweakProcessing0 = CompilerOptions.TweakProcessing.STRIP;
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setTweakProcessing(compilerOptions_TweakProcessing0);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      List<String> list0 = ResourceBundle.Control.FORMAT_CLASS;
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setTweak(list0);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      SourceMap.DetailLevel sourceMap_DetailLevel0 = SourceMap.DetailLevel.ALL;
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setSourceMapDetailLevel(sourceMap_DetailLevel0);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      SourceMap.Format sourceMap_Format0 = SourceMap.Format.DEFAULT;
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setSourceMapFormat(sourceMap_Format0);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.jvascript.jscmp.CheckPrpertyOrder";
      stringArray0[1] = "com.google.jvascript.jscmp.CheckPrpertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Function<Integer, Boolean> function0 = (Function<Integer, Boolean>) mock(Function.class, new ViolatedAssumptionAnswer());
      // Undeclared exception!
      try { 
        commandLineRunner0.enableTestMode((Supplier<List<JSSourceFile>>) null, (Supplier<List<JSSourceFile>>) null, (Supplier<List<JSModule>>) null, function0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "kK";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Supplier<List<JSSourceFile>> supplier0 = (Supplier<List<JSSourceFile>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      Supplier<List<JSModule>> supplier1 = (Supplier<List<JSModule>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      Function<Integer, Boolean> function0 = (Function<Integer, Boolean>) mock(Function.class, new ViolatedAssumptionAnswer());
      // Undeclared exception!
      try { 
        commandLineRunner0.enableTestMode(supplier0, supplier0, supplier1, function0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.scomp.CheckPropertyOrder";
      stringArray0[1] = "kK";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      commandLineRunner0.setRunOptions(compilerOptions0);
      assertFalse(compilerOptions0.removeEmptyFunctions);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "kK";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      List<String> list0 = ResourceBundle.Control.FORMAT_DEFAULT;
      List<JSSourceFile> list1 = commandLineRunner0.createInputs(list0, false);
      assertFalse(list1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "coQ.googl@.javascrip.jscomp.CheckPropertyOrder";
      stringArray0[1] = "coQ.googl@.javascrip.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      // Undeclared exception!
      try { 
        commandLineRunner0.createJsModules((List<String>) null, (List<String>) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "?om.google.jav/script.scomEFCheckPropertyOrder";
      stringArray0[1] = "kK";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Vector<String> vector0 = new Vector<String>();
      // Undeclared exception!
      try { 
        commandLineRunner0.createJsModules(vector0, vector0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "com.google.javascript.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Vector<String> vector0 = new Vector<String>();
      vector0.add(";:888");
      try { 
        commandLineRunner0.createJsModules(vector0, vector0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Invalid module name: ';'
         //
         verifyException("com.google.javascript.jscomp.AbstractCommandLineRunner", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "kK";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      commandLineRunner0.checkModuleName("Mmi82PaC");
      assertFalse(commandLineRunner0.shouldRunCompiler());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Stack<JSModule> stack0 = new Stack<JSModule>();
      // Undeclared exception!
      try { 
        AbstractCommandLineRunner.parseModuleWrappers((List<String>) null, stack0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      JSModule jSModule0 = new JSModule(";Q8*8");
      LinkedList<JSModule> linkedList0 = new LinkedList<JSModule>();
      linkedList0.add(jSModule0);
      List<String> list0 = ResourceBundle.Control.FORMAT_CLASS;
      try { 
        AbstractCommandLineRunner.parseModuleWrappers(list0, linkedList0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Expected module wrapper to have <name>:<wrapper> format: java.class
         //
         verifyException("com.google.javascript.jscomp.AbstractCommandLineRunner", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      LinkedList<String> linkedList0 = new LinkedList<String>();
      LinkedList<JSModule> linkedList1 = new LinkedList<JSModule>();
      Map<String, String> map0 = AbstractCommandLineRunner.parseModuleWrappers(linkedList0, linkedList1);
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Vector<String> vector0 = new Vector<String>();
      vector0.add(";:888");
      LinkedList<JSModule> linkedList0 = new LinkedList<JSModule>();
      try { 
        AbstractCommandLineRunner.parseModuleWrappers(vector0, linkedList0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unknown module: ';'
         //
         verifyException("com.google.javascript.jscomp.AbstractCommandLineRunner", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder(1092);
      Compiler compiler0 = new Compiler();
      AbstractCommandLineRunner.writeOutput(stringBuilder0, compiler0, "Bad --externs flag. ", "com.google.javascript.jscomp.Concrete[ype$7", "%MJe^L@cO'E(jCY`3}");
      assertEquals("Bad --externs flag. \n", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder(1092);
      Compiler compiler0 = new Compiler();
      AbstractCommandLineRunner.writeOutput(stringBuilder0, compiler0, "com.google.javascript.jscomp.Concrete[ype$7", "minimizeExitPoints", "minimizeExitPoints");
      assertEquals("com.google.javascript.jscomp.Concrete[ype$7\n", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder(1092);
      AbstractCommandLineRunner.writeOutput(stringBuilder0, (Compiler) null, "N5AvmbEV|%*P-v88x", "com.google.javascript.jscomp.Concrete[ype$7", "com.google.javascript.jscomp.Concrete[ype$7");
      assertEquals("N5AvmbEV|%*P-v88x\n", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "com.google.javascript.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Compiler compiler0 = commandLineRunner0.createCompiler();
      ArrayList<JSSourceFile> arrayList0 = new ArrayList<JSSourceFile>();
      Vector<JSModule> vector0 = new Vector<JSModule>(1834, (-2045529009));
      Result result0 = compiler0.compileModules(arrayList0, vector0, compilerOptions0);
      int int0 = commandLineRunner0.processResults(result0, vector0, compilerOptions0);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "com.google.javascript.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      String string0 = commandLineRunner0.expandManifest((JSModule) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "com.google.javascript.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      OutputStream outputStream0 = commandLineRunner0.filenameToOutputStream("Lfy*S&>0Jrs_r");
      assertNotNull(outputStream0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.javascript.jscomp.CheckPropertyOrder";
      stringArray0[1] = "com.google.javascript.jscomp.CheckPropertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      OutputStream outputStream0 = commandLineRunner0.filenameToOutputStream((String) null);
      assertNull(outputStream0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "com.google.jvascript.jscmp.CheckPrpertyOrder";
      stringArray0[1] = "com.google.jvascript.jscmp.CheckPrpertyOrder";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      MockPrintStream mockPrintStream0 = new MockPrintStream("com.google.jvascript.jscmp.CheckPrpertyOrder");
      ArrayList<JSModule> arrayList0 = new ArrayList<JSModule>();
      JSModuleGraph jSModuleGraph0 = new JSModuleGraph(arrayList0);
      commandLineRunner0.printModuleGraphManifestTo(jSModuleGraph0, mockPrintStream0);
      assertFalse(commandLineRunner0.shouldRunCompiler());
  }
}
