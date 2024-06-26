/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:24:12 GMT 2023
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
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.Result;
import com.google.javascript.jscomp.SourceMap;
import java.io.CharArrayWriter;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.ResourceBundle;
import java.util.Stack;
import java.util.Vector;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.System;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileWriter;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AbstractCommandLineRunner_ESTest extends AbstractCommandLineRunner_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      String[] stringArray0 = new String[7];
      stringArray0[0] = "Q/ %ql4R1lQY>";
      stringArray0[1] = "fsH";
      stringArray0[2] = "";
      stringArray0[3] = "";
      stringArray0[4] = "com.google.javascript.jscomp.TypedScopeCreator$GlobalScopeBuilder";
      stringArray0[5] = "";
      stringArray0[6] = "";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Supplier<List<JSSourceFile>> supplier0 = (Supplier<List<JSSourceFile>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(supplier0).get();
      Supplier<List<JSModule>> supplier1 = (Supplier<List<JSModule>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      Function<Integer, Boolean> function0 = (Function<Integer, Boolean>) mock(Function.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(function0).apply(anyInt());
      commandLineRunner0.enableTestMode(supplier0, (Supplier<List<JSSourceFile>>) null, supplier1, function0);
      commandLineRunner0.run();
      assertFalse(commandLineRunner0.shouldRunCompiler());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String[] stringArray0 = new String[0];
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
  public void test02()  throws Throwable  {
      String[] stringArray0 = new String[1];
      stringArray0[0] = "^a+";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      commandLineRunner0.initOptionsFromFlags(compilerOptions0);
      assertFalse(compilerOptions0.removeEmptyFunctions);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String[] stringArray0 = new String[9];
      stringArray0[0] = "ECMASCRIPT3";
      stringArray0[1] = "bCW";
      stringArray0[2] = "";
      stringArray0[3] = "X/%0!xOBptw&t\"*R";
      stringArray0[4] = "E?c+";
      stringArray0[5] = "com.google.javascript.jscomp.ProcessTweaks$TweakInfo";
      stringArray0[6] = "I\"";
      stringArray0[7] = "o-~";
      stringArray0[8] = "ECMASCRIPT5";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Compiler compiler0 = commandLineRunner0.getCompiler();
      assertNull(compiler0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      CompilerOptions.TweakProcessing compilerOptions_TweakProcessing0 = CompilerOptions.TweakProcessing.STRIP;
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setTweakProcessing(compilerOptions_TweakProcessing0);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Stack<String> stack0 = new Stack<String>();
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setTweak(stack0);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "^a+";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setSourceMapDetailLevel(compilerOptions0.sourceMapDetailLevel);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      SourceMap.Format sourceMap_Format0 = SourceMap.Format.EXPERIMENTIAL;
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setSourceMapFormat(sourceMap_Format0);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Supplier<List<JSSourceFile>> supplier0 = (Supplier<List<JSSourceFile>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      Supplier<List<JSSourceFile>> supplier1 = (Supplier<List<JSSourceFile>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      Supplier<List<JSModule>> supplier2 = (Supplier<List<JSModule>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      Function<Integer, Boolean> function0 = (Function<Integer, Boolean>) mock(Function.class, new ViolatedAssumptionAnswer());
      // Undeclared exception!
      try { 
        commandLineRunner0.enableTestMode(supplier0, supplier1, supplier2, function0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String[] stringArray0 = new String[2];
      stringArray0[0] = "fsH";
      stringArray0[1] = "fsH";
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
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      commandLineRunner0.setRunOptions(compilerOptions0);
      assertFalse(compilerOptions0.ideMode);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String[] stringArray0 = new String[0];
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
  public void test12()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Vector<String> vector0 = new Vector<String>();
      List<JSSourceFile> list0 = commandLineRunner0.createInputs(vector0, false);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      String[] stringArray0 = new String[7];
      stringArray0[0] = "Q/ %ql4R1lQY>";
      stringArray0[1] = "fsH";
      stringArray0[2] = "";
      stringArray0[3] = "";
      stringArray0[4] = "com.google.javascript.jscomp.TypedScopeCreator$GlobalScopeBuilder";
      stringArray0[5] = "";
      stringArray0[6] = "";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Supplier<List<JSSourceFile>> supplier0 = (Supplier<List<JSSourceFile>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      LinkedList<JSModule> linkedList0 = new LinkedList<JSModule>();
      Supplier<List<JSModule>> supplier1 = (Supplier<List<JSModule>>) mock(Supplier.class, new ViolatedAssumptionAnswer());
      doReturn(linkedList0).when(supplier1).get();
      Function<Integer, Boolean> function0 = (Function<Integer, Boolean>) mock(Function.class, new ViolatedAssumptionAnswer());
      commandLineRunner0.enableTestMode(supplier0, (Supplier<List<JSSourceFile>>) null, supplier1, function0);
      List<String> list0 = ResourceBundle.Control.FORMAT_CLASS;
      List<JSModule> list1 = commandLineRunner0.createJsModules(list0, list0);
      assertEquals(0, list1.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String[] stringArray0 = new String[0];
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
  public void test15()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      List<String> list0 = ResourceBundle.Control.FORMAT_DEFAULT;
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      ArrayList<Locale.LanguageRange> arrayList0 = new ArrayList<Locale.LanguageRange>();
      Locale.FilteringMode locale_FilteringMode0 = Locale.FilteringMode.IGNORE_EXTENDED_RANGES;
      List<String> list1 = Locale.filterTags((List<Locale.LanguageRange>) arrayList0, (Collection<String>) compilerOptions0.stripTypes, locale_FilteringMode0);
      // Undeclared exception!
      try { 
        commandLineRunner0.createJsModules(list1, list0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      List<String> list0 = ResourceBundle.Control.FORMAT_CLASS;
      // Undeclared exception!
      try { 
        commandLineRunner0.createJsModules(list0, (List<String>) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      commandLineRunner0.checkModuleName("NyNdA");
      assertTrue(commandLineRunner0.shouldRunCompiler());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      try { 
        commandLineRunner0.checkModuleName("--tweak flag syntax invalid: ");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Invalid module name: '--tweak flag syntax invalid: '
         //
         verifyException("com.google.javascript.jscomp.AbstractCommandLineRunner", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
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
  public void test20()  throws Throwable  {
      ArrayList<String> arrayList0 = new ArrayList<String>();
      Stack<JSModule> stack0 = new Stack<JSModule>();
      Map<String, String> map0 = AbstractCommandLineRunner.parseModuleWrappers(arrayList0, stack0);
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      List<String> list0 = ResourceBundle.Control.FORMAT_DEFAULT;
      LinkedList<JSModule> linkedList0 = new LinkedList<JSModule>();
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
  public void test22()  throws Throwable  {
      Logger logger0 = Logger.getAnonymousLogger();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("){");
      AbstractCommandLineRunner.writeOutput(mockPrintWriter0, compiler0, "?~J", "!u0O", "bf'/mF|e4Xa,}TdXQ");
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      Logger logger0 = Logger.getAnonymousLogger();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      AbstractCommandLineRunner.writeOutput(charArrayWriter0, compiler0, "&H?O04we\"%Irg9", "bf'/mF|e4Xa,}TdXQ", "bf'/mF|e4Xa,}TdXQ");
      assertEquals(15, charArrayWriter0.size());
      assertEquals("&H?O04we\"%Irg9\n", charArrayWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      MockFile mockFile0 = new MockFile("", "yLygq-I\"g");
      MockFileWriter mockFileWriter0 = new MockFileWriter(mockFile0, false);
      Logger logger0 = Logger.getAnonymousLogger();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      AbstractCommandLineRunner.writeOutput(mockFileWriter0, compiler0, ",", "yLygq-I\"g", "");
      assertEquals(0, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      Compiler compiler0 = commandLineRunner0.createCompiler();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[1];
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode((String) null, (String) null);
      jSSourceFileArray0[0] = jSSourceFile0;
      Result result0 = compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      int int0 = commandLineRunner0.processResults(result0, (List<JSModule>) null, compilerOptions0);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      String string0 = commandLineRunner0.expandSourceMapPath(compilerOptions0, (JSModule) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      String string0 = commandLineRunner0.expandManifest((JSModule) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      OutputStream outputStream0 = commandLineRunner0.filenameToOutputStream("ECMASCRIPT3");
      assertNotNull(outputStream0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      OutputStream outputStream0 = commandLineRunner0.filenameToOutputStream((String) null);
      assertNull(outputStream0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      List<String> list0 = ResourceBundle.Control.FORMAT_DEFAULT;
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      AbstractCommandLineRunner.createDefineOrTweakReplacements(list0, compilerOptions0, false);
      assertFalse(compilerOptions0.checkUnusedPropertiesEarly);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      String[] stringArray0 = new String[0];
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      List<String> list0 = ResourceBundle.Control.FORMAT_DEFAULT;
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      AbstractCommandLineRunner.createDefineOrTweakReplacements(list0, compilerOptions0, true);
      assertFalse(compilerOptions0.prettyPrint);
  }
}
