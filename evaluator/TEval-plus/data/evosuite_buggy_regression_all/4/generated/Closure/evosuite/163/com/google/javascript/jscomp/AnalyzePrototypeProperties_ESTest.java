/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:25:05 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AnalyzePrototypeProperties;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSModuleGraph;
import com.google.javascript.rhino.Node;
import java.util.Collection;
import java.util.Deque;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AnalyzePrototypeProperties_ESTest extends AnalyzePrototypeProperties_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, false, false);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.ShadowVariables$GatherReferenceInfo");
      analyzePrototypeProperties0.process(node0, node0);
      assertEquals(29, Node.JSDOC_INFO_PROP);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, true, true);
      AnalyzePrototypeProperties.NameInfo analyzePrototypeProperties_NameInfo0 = analyzePrototypeProperties0.new NameInfo("");
      Deque<AnalyzePrototypeProperties.Symbol> deque0 = analyzePrototypeProperties_NameInfo0.getDeclarations();
      assertEquals(0, deque0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, false, false);
      AnalyzePrototypeProperties.NameInfo analyzePrototypeProperties_NameInfo0 = analyzePrototypeProperties0.new NameInfo("");
      String string0 = analyzePrototypeProperties_NameInfo0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, false, false);
      AnalyzePrototypeProperties.NameInfo analyzePrototypeProperties_NameInfo0 = analyzePrototypeProperties0.new NameInfo("");
      boolean boolean0 = analyzePrototypeProperties_NameInfo0.readsClosureVariables();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, false, false);
      Collection<AnalyzePrototypeProperties.NameInfo> collection0 = analyzePrototypeProperties0.getAllNameInfo();
      assertNotNull(collection0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      JSModule jSModule0 = new JSModule("");
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
      JSModule jSModule0 = new JSModule("!#:2RVW");
      AnalyzePrototypeProperties.LiteralProperty analyzePrototypeProperties_LiteralProperty0 = new AnalyzePrototypeProperties.LiteralProperty((Node) null, (Node) null, (Node) null, (Node) null, jSModule0);
      Node node0 = analyzePrototypeProperties_LiteralProperty0.getValue();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      JSModule jSModule0 = new JSModule("com.google.javascript.jscomp.OptimizeCalls");
      AnalyzePrototypeProperties.LiteralProperty analyzePrototypeProperties_LiteralProperty0 = new AnalyzePrototypeProperties.LiteralProperty((Node) null, (Node) null, (Node) null, (Node) null, jSModule0);
      // Undeclared exception!
      try { 
        analyzePrototypeProperties_LiteralProperty0.remove();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.AnalyzePrototypeProperties$LiteralProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      AnalyzePrototypeProperties.LiteralProperty analyzePrototypeProperties_LiteralProperty0 = new AnalyzePrototypeProperties.LiteralProperty((Node) null, (Node) null, (Node) null, (Node) null, (JSModule) null);
      JSModule jSModule0 = analyzePrototypeProperties_LiteralProperty0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      AnalyzePrototypeProperties.AssignmentProperty analyzePrototypeProperties_AssignmentProperty0 = new AnalyzePrototypeProperties.AssignmentProperty((Node) null, (JSModule) null);
      JSModule jSModule0 = analyzePrototypeProperties_AssignmentProperty0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JSModule jSModule0 = new JSModule((String) null);
      AnalyzePrototypeProperties.AssignmentProperty analyzePrototypeProperties_AssignmentProperty0 = new AnalyzePrototypeProperties.AssignmentProperty((Node) null, jSModule0);
      // Undeclared exception!
      try { 
        analyzePrototypeProperties_AssignmentProperty0.remove();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.AnalyzePrototypeProperties$AssignmentProperty", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Node node0 = new Node(1419);
      JSModule jSModule0 = new JSModule("prototype");
      AnalyzePrototypeProperties.AssignmentProperty analyzePrototypeProperties_AssignmentProperty0 = new AnalyzePrototypeProperties.AssignmentProperty(node0, jSModule0);
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
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, true, true);
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.PureFunctionIdentifier");
      analyzePrototypeProperties0.process(node0, node0);
      assertEquals(32, Node.INCRDECR_PROP);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, false, false);
      Node node0 = compiler0.parseTestCode("vfni=xg");
      analyzePrototypeProperties0.process(node0, node0);
      assertEquals(39, Node.EMPTY_BLOCK);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, (JSModuleGraph) null, true, true);
      Node node0 = Node.newString("valueOf");
      AnalyzePrototypeProperties.GlobalFunction analyzePrototypeProperties_GlobalFunction0 = null;
      try {
        analyzePrototypeProperties_GlobalFunction0 = analyzePrototypeProperties0.new GlobalFunction(node0, node0, node0, (JSModule) null);
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
      JSModule[] jSModuleArray0 = new JSModule[1];
      JSModule jSModule0 = new JSModule("Ps}RW*m_(<w.");
      jSModuleArray0[0] = jSModule0;
      JSModuleGraph jSModuleGraph0 = new JSModuleGraph(jSModuleArray0);
      Node node0 = compiler0.parseSyntheticCode("%BrrZ`3FL(*d", "toString");
      AnalyzePrototypeProperties analyzePrototypeProperties0 = new AnalyzePrototypeProperties(compiler0, jSModuleGraph0, false, true);
      analyzePrototypeProperties0.process(node0, node0);
      analyzePrototypeProperties0.process(node0, node0);
      assertFalse(node0.wasEmptyNode());
  }
}