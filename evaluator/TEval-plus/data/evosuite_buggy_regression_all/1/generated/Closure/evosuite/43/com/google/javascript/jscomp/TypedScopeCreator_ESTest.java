/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:07:03 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JqueryCodingConvention;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("deprecated", "deprecated");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscop.DeaultPassConi$20");
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node1 = new Node(120, node0, node0, node0, node0, 41, 51);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(CATCH):  [testcode] :41:51
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscop.DeaultPassConig$20", "com.google.javascript.jscop.DeaultPassConig$20");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("com.google.javascript.jscomp.graph.LinkedDirectedGraph$AnnotatedLinkedDirectedGraphEdge", "com.google.javascript.jscomp.graph.LinkedDirectedGraph$AnnotatedLinkedDirectedGraphEdge");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("dC=unt", "dC=unt");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: consistentIdGenerator
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ActiveXObject");
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node1 = new Node(39, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscop.DeaultPassConi$20");
      Node node1 = new Node(41, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ActiveXObject");
      Node node1 = new Node(43, node0, node0, node0, node0, 32, 1);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscop.DeaultPassConig$20");
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node1 = new Node(44, node0, node0, node0, node0, 54, 44);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("el@kTR5,/QX<b");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(47, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ActiveXObject");
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node1 = new Node(64, node0, node0, node0, node0, 50, 0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("ActiveXObject");
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node1 = new Node(122, node0, node0, node0, node0, 37, 38);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascgipt.jscomp.ReplaceStrings");
      node0.addSuppression("com.google.javascgipt.jscomp.ReplaceStrings");
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node1 = new Node(118, node0, node0, node0, node0, 32, 41);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR):  [testcode] :32:41
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascgipt.jscomp.ReplaceStrings");
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      Node node1 = new Node(118, node0, node0, node0, node0, 32, 41);
      node1.addSuppression("XdiU,r(!cA");
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR):  [testcode] :32:41
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("com.google.javascript.jscop.DeaultPassConi$20");
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscop.DeaultPassConi$20", "com.google.javascript.jscop.DeaultPassConi$20");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.ReplaceStrings");
      Node node1 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.javascript.jscomp.ReplaceStrings", "com.google.javascript.jscomp.ReplaceStrings");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, (CodingConvention) null);
      Node node2 = new Node(51, node1, node0, node0, node0, 32, 38);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TypedScopeCreator", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javas4ript.jscop.DeaultPassCon$20");
      JqueryCodingConvention jqueryCodingConvention0 = new JqueryCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, jqueryCodingConvention0);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      Node node2 = new Node(1, node1, node1, node1, node0, 47, 53);
      Scope scope0 = typedScopeCreator0.createInitialScope(node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}