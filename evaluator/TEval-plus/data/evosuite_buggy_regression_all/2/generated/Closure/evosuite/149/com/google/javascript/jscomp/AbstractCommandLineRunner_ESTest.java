/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:38:28 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCommandLineRunner;
import com.google.javascript.jscomp.CommandLineRunner;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.FunctionInformationMap;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSModuleGraph;
import com.google.javascript.jscomp.PhaseOptimizer;
import com.google.javascript.jscomp.Result;
import com.google.javascript.jscomp.SourceMap;
import com.google.javascript.jscomp.VariableMap;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.ExtensionRegistryLite;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.PushbackInputStream;
import java.io.SequenceInputStream;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.ResourceBundle;
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
      StringBuilder stringBuilder0 = new StringBuilder();
      Compiler compiler0 = commandLineRunner0.createCompiler();
      AbstractCommandLineRunner.writeOutput(stringBuilder0, compiler0, "", "", "");
      assertEquals("\n", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      String[] stringArray0 = new String[1];
      stringArray0[0] = ":&~#";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      JSError[] jSErrorArray0 = new JSError[2];
      HashMap<String, String> hashMap0 = new HashMap<String, String>();
      VariableMap variableMap0 = new VariableMap(hashMap0);
      Enumeration<InputStream> enumeration0 = (Enumeration<InputStream>) mock(Enumeration.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(enumeration0).hasMoreElements();
      SequenceInputStream sequenceInputStream0 = new SequenceInputStream(enumeration0);
      PushbackInputStream pushbackInputStream0 = new PushbackInputStream(sequenceInputStream0);
      ExtensionRegistry extensionRegistry0 = ExtensionRegistry.newInstance();
      FunctionInformationMap functionInformationMap0 = FunctionInformationMap.parseFrom((InputStream) pushbackInputStream0, (ExtensionRegistryLite) extensionRegistry0);
      SourceMap sourceMap0 = new SourceMap();
      Result result0 = new Result(jSErrorArray0, jSErrorArray0, "", variableMap0, variableMap0, variableMap0, functionInformationMap0, sourceMap0, ":&~#");
      int int0 = commandLineRunner0.processResults(result0, (JSModule[]) null, compilerOptions0);
      assertEquals(2, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String[] stringArray0 = new String[9];
      stringArray0[0] = "Bad --js flag. ";
      stringArray0[1] = "Bad --js flag. ";
      stringArray0[2] = "Bad --js flag. ";
      stringArray0[3] = "Bad --js flag. ";
      stringArray0[4] = "8E)lbCCA?";
      stringArray0[5] = "Bad --js flag. ";
      stringArray0[6] = ":DlC";
      stringArray0[7] = "Bad --js flag. ";
      stringArray0[8] = "D@%F;^{";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      Compiler compiler0 = commandLineRunner0.getCompiler();
      assertNull(compiler0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String[] stringArray0 = new String[1];
      stringArray0[0] = ":&~#";
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
      String[] stringArray0 = new String[1];
      stringArray0[0] = "U`R'~";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      commandLineRunner0.setRunOptions(compilerOptions0);
      assertFalse(compilerOptions0.checkTypedPropertyCalls);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Vector<String> vector0 = new Vector<String>();
      vector0.add(":&~#");
      try { 
        AbstractCommandLineRunner.createJsModules(vector0, vector0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Invalid module name: ''
         //
         verifyException("com.google.javascript.jscomp.AbstractCommandLineRunner", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig0 = new AbstractCommandLineRunner.CommandLineConfig();
      SourceMap.DetailLevel sourceMap_DetailLevel0 = SourceMap.DetailLevel.ALL;
      AbstractCommandLineRunner.CommandLineConfig abstractCommandLineRunner_CommandLineConfig1 = abstractCommandLineRunner_CommandLineConfig0.setSourceMapDetailLevel(sourceMap_DetailLevel0);
      assertNotNull(abstractCommandLineRunner_CommandLineConfig1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        AbstractCommandLineRunner.createJsModules((List<String>) null, (List<String>) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Vector<String> vector0 = new Vector<String>();
      // Undeclared exception!
      try { 
        AbstractCommandLineRunner.createJsModules(vector0, vector0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Vector<String> vector0 = new Vector<String>();
      vector0.add("");
      try { 
        AbstractCommandLineRunner.createJsModules(vector0, vector0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Expected 2-4 colon-delimited parts in module spec: 
         //
         verifyException("com.google.javascript.jscomp.AbstractCommandLineRunner", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JSModule[] jSModuleArray0 = new JSModule[0];
      // Undeclared exception!
      try { 
        AbstractCommandLineRunner.parseModuleWrappers((List<String>) null, jSModuleArray0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      JSModule jSModule0 = new JSModule("externsValidation");
      List<String> list0 = PhaseOptimizer.OPTIMAL_ORDER;
      JSModule[] jSModuleArray0 = new JSModule[4];
      jSModuleArray0[0] = jSModule0;
      jSModuleArray0[1] = jSModule0;
      jSModuleArray0[2] = jSModule0;
      jSModuleArray0[3] = jSModule0;
      try { 
        AbstractCommandLineRunner.parseModuleWrappers(list0, jSModuleArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Expected module wrapper to have <name>:<wrapper> format: removeUnreachableCode
         //
         verifyException("com.google.javascript.jscomp.AbstractCommandLineRunner", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      String[] stringArray0 = new String[1];
      stringArray0[0] = "U`R'~";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      StringBuilder stringBuilder0 = new StringBuilder();
      Compiler compiler0 = commandLineRunner0.createCompiler();
      AbstractCommandLineRunner.writeOutput(stringBuilder0, compiler0, "", "", "msg.change.writable.false.to.true.with.configurable.false");
      assertEquals("\n", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      String[] stringArray0 = new String[1];
      stringArray0[0] = ":&~#";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      CompilerOptions compilerOptions0 = commandLineRunner0.createOptions();
      JSModule jSModule0 = new JSModule("externsValidation");
      String string0 = commandLineRunner0.expandSourceMapPath(compilerOptions0, jSModule0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      String[] stringArray0 = new String[1];
      stringArray0[0] = "U`R'~";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      JSModule jSModule0 = new JSModule((String) null);
      String string0 = commandLineRunner0.expandManifest(jSModule0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      List<String> list0 = ResourceBundle.Control.FORMAT_CLASS;
      CompilerOptions compilerOptions0 = new CompilerOptions();
      AbstractCommandLineRunner.createDefineReplacements(list0, compilerOptions0);
      assertFalse(compilerOptions0.instrumentForCoverageOnly);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String[] stringArray0 = new String[1];
      stringArray0[0] = "U`R'~";
      CommandLineRunner commandLineRunner0 = new CommandLineRunner(stringArray0);
      LinkedList<JSModule> linkedList0 = new LinkedList<JSModule>();
      JSModuleGraph jSModuleGraph0 = new JSModuleGraph(linkedList0);
      PipedInputStream pipedInputStream0 = new PipedInputStream(164);
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream(pipedInputStream0);
      ObjectOutputStream objectOutputStream0 = new ObjectOutputStream(pipedOutputStream0);
      MockPrintStream mockPrintStream0 = new MockPrintStream(objectOutputStream0);
      commandLineRunner0.printModuleGraphManifestTo(jSModuleGraph0, mockPrintStream0);
      assertFalse(commandLineRunner0.shouldRunCompiler());
  }
}