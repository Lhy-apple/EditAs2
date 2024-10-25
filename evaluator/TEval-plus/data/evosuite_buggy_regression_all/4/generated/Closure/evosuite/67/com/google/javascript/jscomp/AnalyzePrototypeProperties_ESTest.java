/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:14:46 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.Multimap;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.AnalyzePrototypeProperties;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSModuleGraph;
import com.google.javascript.jscomp.LightweightMessageFormatter;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.PrintStreamErrorManager;
import com.google.javascript.jscomp.SyntheticAst;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AnalyzePrototypeProperties_ESTest extends AnalyzePrototypeProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "function JSCompiler_get(JSCompiler_get_name) {  return function() {return this[JSCompiler_get_name]}}", "function JSCompiler_get(JSCompiler_get_name) {  return function() {return this[JSCompiler_get_name]}}");
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, false, false);
      analyzePrototypeProperties0.process(node0, node0);
      assertEquals(12, Node.COLUMN_BITS);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "function JSCompiler_get(JSCompiler_get_name) {  return function() {return this[JSCompile_get_nae]}}", "function JSCompiler_get(JSCompiler_get_name) {  return function() {return this[JSCompile_get_nae]}}");
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, true, true);
      analyzePrototypeProperties0.process(node0, node0);
      assertEquals(2, Node.SPECIALCALL_WITH);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties((AbstractCompiler) null, (JSModuleGraph) null, false, false);
      AnalyzePrototypeProperties.NameInfo analyzePrototypeProperties_NameInfo0 = analyzePrototypeProperties0.new NameInfo((String) null);
      String string0 = analyzePrototypeProperties_NameInfo0.toString();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, true, true);
      AnalyzePrototypeProperties.NameInfo analyzePrototypeProperties_NameInfo0 = analyzePrototypeProperties0.new NameInfo((String) null);
      boolean boolean0 = analyzePrototypeProperties_NameInfo0.readsClosureVariables();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, true, true);
      Collection<AnalyzePrototypeProperties.NameInfo> collection0 = analyzePrototypeProperties0.getAllNameInfo();
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JSModule jSModule0 = new JSModule("valueOf");
      AnalyzePrototypeProperties.LiteralProperty analyzePrototypeProperties_LiteralProperty0 = new AnalyzePrototypeProperties.LiteralProperty((Node) null, (Node) null, (Node) null, (Node) null, jSModule0);
      // Undeclared exception!
      try { 
        analyzePrototypeProperties_LiteralProperty0.getPrototype();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.AnalyzePrototypeProperties$LiteralProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      AnalyzePrototypeProperties.LiteralProperty analyzePrototypeProperties_LiteralProperty0 = new AnalyzePrototypeProperties.LiteralProperty((Node) null, (Node) null, (Node) null, (Node) null, (JSModule) null);
      Node node0 = analyzePrototypeProperties_LiteralProperty0.getValue();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = Node.newNumber((-2001.4356678557829), 2920, 2920);
      JSModule jSModule0 = new JSModule("6(l[yrM&YG");
      AnalyzePrototypeProperties.LiteralProperty analyzePrototypeProperties_LiteralProperty0 = new AnalyzePrototypeProperties.LiteralProperty(node0, node0, node0, node0, jSModule0);
      // Undeclared exception!
      try { 
        analyzePrototypeProperties_LiteralProperty0.remove();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.rhino.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LightweightMessageFormatter lightweightMessageFormatter0 = LightweightMessageFormatter.withoutSource();
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(lightweightMessageFormatter0, (PrintStream) null);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      SyntheticAst syntheticAst0 = new SyntheticAst("$VALUES");
      Node node0 = syntheticAst0.getAstRoot(compiler0);
      AnalyzePrototypeProperties.LiteralProperty analyzePrototypeProperties_LiteralProperty0 = new AnalyzePrototypeProperties.LiteralProperty(node0, node0, node0, node0, (JSModule) null);
      JSModule jSModule0 = analyzePrototypeProperties_LiteralProperty0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "", "v08X4Tg9tkFt?MmPOF");
      AnalyzePrototypeProperties.AssignmentProperty analyzePrototypeProperties_AssignmentProperty0 = new AnalyzePrototypeProperties.AssignmentProperty(node0, (JSModule) null);
      // Undeclared exception!
      try { 
        analyzePrototypeProperties_AssignmentProperty0.getPrototype();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.AnalyzePrototypeProperties$AssignmentProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) arrayList0);
      AnalyzePrototypeProperties.AssignmentProperty analyzePrototypeProperties_AssignmentProperty0 = new AnalyzePrototypeProperties.AssignmentProperty(node0, (JSModule) null);
      JSModule jSModule0 = analyzePrototypeProperties_AssignmentProperty0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Node node0 = Node.newNumber(4.294967296E9);
      AnalyzePrototypeProperties.AssignmentProperty analyzePrototypeProperties_AssignmentProperty0 = new AnalyzePrototypeProperties.AssignmentProperty(node0, (JSModule) null);
      // Undeclared exception!
      try { 
        analyzePrototypeProperties_AssignmentProperty0.remove();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JSModule jSModule0 = new JSModule("SEE");
      AnalyzePrototypeProperties.AssignmentProperty analyzePrototypeProperties_AssignmentProperty0 = new AnalyzePrototypeProperties.AssignmentProperty((Node) null, jSModule0);
      // Undeclared exception!
      try { 
        analyzePrototypeProperties_AssignmentProperty0.getValue();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.AnalyzePrototypeProperties$AssignmentProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.jscomp.AnalyzePrototypeProperties$LiteralProperty", "com.google.javascript.jscomp.AnalyzePrototypeProperties$LiteralProperty");
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, false, false);
      analyzePrototypeProperties0.process(node0, node0);
      assertFalse(node0.isOptionalArg());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("fF5'!_#(x'K5 {7m");
      JSModule jSModule0 = new JSModule((String) null);
      JSModule[] jSModuleArray0 = new JSModule[1];
      jSModuleArray0[0] = jSModule0;
      JSModuleGraph jSModuleGraph0 = new JSModuleGraph(jSModuleArray0);
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, jSModuleGraph0, true, true);
      AnalyzePrototypeProperties.GlobalFunction analyzePrototypeProperties_GlobalFunction0 = null;
      try {
        analyzePrototypeProperties_GlobalFunction0 = analyzePrototypeProperties0.new GlobalFunction(node0, node0, node0, jSModule0);
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
      Compiler compiler0 = new Compiler();
      Integer integer0 = new Integer((-1854));
      JSModule jSModule0 = new JSModule("9");
      Integer integer1 = new Integer((-1789569705));
      ImmutableMultimap<Integer, JSModule> immutableMultimap0 = ImmutableMultimap.of(integer0, jSModule0, integer1, jSModule0, integer1, jSModule0, integer1, jSModule0);
      LinkedListMultimap<Integer, JSModule> linkedListMultimap0 = LinkedListMultimap.create((Multimap<? extends Integer, ? extends JSModule>) immutableMultimap0);
      LinkedList<JSModule> linkedList0 = new LinkedList<JSModule>();
      Vector<JSModule> vector0 = new Vector<JSModule>(linkedList0);
      List<JSModule> list0 = linkedListMultimap0.replaceValues(integer0, vector0);
      JSModuleGraph jSModuleGraph0 = new JSModuleGraph(list0);
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, jSModuleGraph0, true, false);
      AnalyzePrototypeProperties.NameInfo analyzePrototypeProperties_NameInfo0 = analyzePrototypeProperties0.new NameInfo("9");
      boolean boolean0 = analyzePrototypeProperties_NameInfo0.markReference(jSModule0);
      assertTrue(boolean0);
      
      boolean boolean1 = analyzePrototypeProperties_NameInfo0.markReference(jSModule0);
      assertFalse(boolean1);
  }
}
