/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:52:30 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.TightenTypes;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.common.base.Platform$1", "com.google.common.base.Platform$1");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(37, node0, node0, 16, 1);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MockFile mockFile0 = new MockFile(" |{!G8EAr");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile((File) mockFile0);
      Node node0 = compiler0.parse((SourceFile) jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(120, node0, node0, node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(CATCH):  |{!G8EAr:-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("Constructor expected as first argument");
      Node node0 = compiler0.parse((SourceFile) jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(64, node0, node0, node0, 4095, 12);
      Node node2 = new Node(12, node1, node1, node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.common.base.Platform$1", "com.google.common.base.Platform$1");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
      
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "compiler.version", "compiler.version");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(118, node0, node0, node0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR): [testcode]:-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "compiler.version", "compiler.version");
      Node node1 = new Node(105, node0, 2171, (-2004));
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(" D{!GdE");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(86, node0, node0, node0, 4095, 29);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("K;bz3tVOo:6");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      typedScopeCreator0.patchGlobalScope(scope0, node0);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "compiler.version", "compiler.version");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(41, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.common.collect.LinkedListMultimap$AsMapEntries");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(43, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MockFile mockFile0 = new MockFile(" |{!G8EAr");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile((File) mockFile0);
      Node node0 = compiler0.parse((SourceFile) jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(44, node0, node0, 46, 1919);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(" D{!GdE");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(47, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(32, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MockFile mockFile0 = new MockFile(" r{!GdE");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile((File) mockFile0);
      Node node0 = compiler0.parse((SourceFile) jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(122, node0, node0, node0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MockFile mockFile0 = new MockFile("r;^/p_}`@wrX*j");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile((File) mockFile0);
      Node node0 = compiler0.parse((SourceFile) jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      node0.addSuppression("'/72^r$Ly8?/kLc8");
      Node node1 = new Node(118, node0, node0, node0, 47, 133);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "compiler.version", "compiler.version");
      node0.setType((-938));
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Node node1 = new Node(118, node0, node0, node0);
      Node node2 = new Node(16, node1, node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MockFile mockFile0 = new MockFile(".2FPYu\"fZg^b?*dMszkw", ".2FPYu\"fZg^b?*dMszkw");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile((File) mockFile0);
      Node node0 = compiler0.parse((SourceFile) jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = Node.newString(".2FPYu\"fZg^b?*dMszkw");
      Node node2 = new Node(118, node0, node1, node1, 12, 4095);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MockFile mockFile0 = new MockFile("[?5T;=*KSle-");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile((File) mockFile0);
      jSSourceFile0.setIsExtern(true);
      Node node0 = compiler0.parse((SourceFile) jSSourceFile0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(118, node0, node0, node0, 47, 132);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "compiler.version", "compiler.version");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Node node1 = Normalize.parseAndNormalizeTestCode(compiler0, "compiler.version", "compiler.version");
      Node node2 = new Node(1, node1, node1, node0, 47, (-2054844731));
      Scope scope1 = typedScopeCreator0.createScope(node2, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.common.base.Platform$1", "com.google.common.base.Platform$1");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>(44);
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) arrayList0);
      Node node2 = new Node(1152, node0, node1, node1);
      Node node3 = new Node((-1017), node2);
      Scope scope1 = typedScopeCreator0.createScope(node3, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "com.google.common.base.Platform$1", "com.google.common.base.Platform$1");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      TightenTypes tightenTypes0 = new TightenTypes(compiler0);
      JSTypeRegistry jSTypeRegistry0 = tightenTypes0.getTypeRegistry();
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node1 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      Node node2 = new Node(53, node0, node0, node1);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node2, scope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 53
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }
}
