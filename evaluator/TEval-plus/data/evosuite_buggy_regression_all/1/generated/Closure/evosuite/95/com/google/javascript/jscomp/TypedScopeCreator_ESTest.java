/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:12:21 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TypedScopeCreator_ESTest extends TypedScopeCreator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("5H7m +qaAvS{", "5H7m +qaAvS{");
      Node node0 = Node.newString(120, "5H7m +qaAvS{");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, (Scope) null);
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
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("com.google.javascript.rhino.jstype.NoType");
      Node node1 = new Node(46, node0, 38, 40);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      assertEquals(1, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("1= +w");
      Node node1 = new Node(118, node0, 44, 28);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR):  [testcode] :44:28
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("TypedScpeCreator$Ab=tza7tScopeBuildr");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("W^o2,0>gD*g7LT8");
      Node node1 = new Node(41, node0, node0, node0, node0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("HnYotctor");
      Node node0 = new Node(43);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("JSC_MALFORMED_TYPEDEF");
      Node node1 = new Node(44, node0, 49, 5);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      compiler0.parseTestCode("g1W= +E");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      Node node0 = Node.newString(47, "g1W= +E");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("TypedScopeCreator$AbstractScopeBuilder");
      Node node0 = Node.newString(64, "TypedScopeCreator$AbstractScopeBuilder");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      typedScopeCreator0.createScope(node0, (Scope) null);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("[objec ");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = new Node(69, node0, (-1147), 12);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(33, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("[object ");
      Node node0 = Node.newString(122, "[object ");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("sypedScopiCreator$Abstra}tScopeBuilder");
      Node node1 = new Node(105, node0, 4, 2);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node2 = new Node(2184, node1);
      Scope scope0 = new Scope(node2, compiler0);
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
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("TypedScpeCreator$Ab=tza7tScopeBuildr");
      Node node1 = new Node(37, node0, 18, 34);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertTrue(scope0.isGlobal());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("qe.xHV");
      Node node1 = new Node(86, node0, 43, 43);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      node1.setJSDocInfo(jSDocInfo0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node1, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      Node node0 = compiler0.parseTestCode("1= +w");
      node0.removeChildren();
      Node node1 = new Node(118, node0, (-2), 28);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node1, (Scope) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(VAR):  [testcode] :-1:-1
         // [source unknown]
         //   Parent: NULL
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("W^o_,0>gD*g7LT8");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = compiler0.parseSyntheticCode("W^o_,0>gD*g7LT8", "com.google.javascript.jscomp.TypedScopeCreator$DiscoverEnums");
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertEquals(34, scope0.getVarCount());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("goog.typedef");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      assertTrue(scope0.isGlobal());
  }
}