/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:32:49 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.LinkedFlowScope;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.JSDocInfo;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.NoType;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.StaticScope;
import com.google.javascript.rhino.jstype.StaticSlot;
import java.util.Iterator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Scope_ESTest extends Scope_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("PL6k:@qX;", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isLocal();
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("lw", (Node) null, (JSType) null, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.isGlobal();
      assertTrue(boolean0);
      assertFalse(scope_Var0.isDefine());
      assertFalse(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("JSC_EXPORTED_FUNCTION_UNKNOWN_PARAMETER_TYPE", (Node) null, (JSType) null, (CompilerInput) null);
      scope_Var0.getJSDocInfo();
      assertFalse(scope_Var0.isDefine());
      assertTrue(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("JSC_EXPORTED_FUNCTION_UNKNOWN_PARAMETER_TYPE", (Node) null, (JSType) null, (CompilerInput) null);
      String string0 = scope_Var0.getName();
      assertTrue(scope_Var0.isTypeInferred());
      assertNotNull(string0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("lw", (Node) null, (JSType) null, (CompilerInput) null, false);
      scope_Var0.getNameNode();
      assertFalse(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = Node.newString("LEAST_FUNCTION_TYPE", (-418), (-3582));
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      NoType noType0 = (NoType)jSTypeRegistry0.getGreatestSubtypeWithProperty((JSType) null, "Named type with empty name component");
      Scope scope0 = new Scope(node0, noType0);
      Scope.Var scope_Var0 = scope0.declare("x", node0, noType0, (CompilerInput) null, false);
      scope_Var0.getScope();
      assertFalse(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("PL6k:@qX;", (Node) null, (JSType) null, (CompilerInput) null);
      scope_Var0.getType();
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("PL6k:@qX;", (Node) null, (JSType) null, (CompilerInput) null);
      assertTrue(scope_Var0.isTypeInferred());
      
      scope_Var0.setType((JSType) null);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("lw", (Node) null, (JSType) null, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.isDefine();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("Y<|NGI!vB+X", (Node) null, (JSType) null, (CompilerInput) null);
      // Undeclared exception!
      try { 
        scope_Var0.isBleedingFunction();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("com.google.javascript.jscomp.Scope", (Node) null, (JSType) null, (CompilerInput) null);
      String string0 = scope_Var0.toString();
      assertFalse(scope_Var0.isDefine());
      assertEquals("Scope.Var com.google.javascript.jscomp.Scope", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Scope scope0 = null;
      try {
        scope0 = new Scope((Node) null, compiler0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      boolean boolean0 = scope0.isBottom();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      StaticSlot<JSType> staticSlot0 = scope0.getSlot((String) null);
      assertNull(staticSlot0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = scope0.getRootNode();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      int int0 = scope0.getDepth();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      int int0 = scope0.getVarCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      JSType jSType0 = linkedFlowScope0.getTypeOfThis();
      assertNull(jSType0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      StaticSlot<JSType> staticSlot0 = scope0.getOwnSlot("XlG6A9AP6wg@md<");
      assertNull(staticSlot0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Iterator<Scope.Var> iterator0 = scope0.getVars();
      assertNotNull(iterator0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      StaticScope<JSType> staticScope0 = scope0.getParentScope();
      assertNull(staticScope0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Node node0 = Node.newString("com.google.javascript.jscomp.Scope", 4095, 2072);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Scope.Var scope_Var0 = scope0.declare("com.google.javascript.jscomp.WarningsGuard", node0, objectType0, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isExtern();
      assertTrue(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      SourceFile sourceFile0 = SourceFile.fromCode("G^,O{EeVf", "G^,O{EeVf");
      JsAst jsAst0 = new JsAst(sourceFile0);
      CompilerInput compilerInput0 = new CompilerInput(jsAst0, false);
      Scope.Var scope_Var0 = scope0.declare("==", (Node) null, (JSType) null, compilerInput0);
      boolean boolean0 = scope_Var0.isExtern();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      SourceFile sourceFile0 = SourceFile.fromCode("t-N*^X5s", "t-N*^X5s");
      JsAst jsAst0 = new JsAst(sourceFile0);
      CompilerInput compilerInput0 = new CompilerInput(jsAst0, true);
      Scope.Var scope_Var0 = scope0.declare("t-N*^X5s", (Node) null, (JSType) null, compilerInput0);
      boolean boolean0 = scope_Var0.isExtern();
      assertFalse(scope_Var0.isDefine());
      assertTrue(boolean0);
      assertEquals("t-N*^X5s", scope_Var0.getInputName());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("Y<|NGI!vB+X", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isConst();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Node node0 = Node.newString(4017, "`tP", 4017, 4017);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("`tP", node0, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isConst();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newNumber((-743.7647200404185));
      SourceFile sourceFile0 = SourceFile.fromCode("]X", "]X");
      JsAst jsAst0 = new JsAst(sourceFile0);
      CompilerInput compilerInput0 = new CompilerInput(jsAst0, false);
      Scope.Var scope_Var0 = scope0.declare("]X", node0, (JSType) null, compilerInput0);
      Node node1 = new Node(19, node0, node0);
      Node node2 = scope_Var0.getInitialValue();
      assertNull(node2);
      assertFalse(scope_Var0.isDefine());
      assertEquals("]X", scope_Var0.getInputName());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Node node0 = Node.newString("LEAST_FUNCTION_TYPE", (-418), (-3582));
      Node node1 = new Node(86, node0, node0, node0);
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      NoType noType0 = (NoType)jSTypeRegistry0.getGreatestSubtypeWithProperty((JSType) null, "Named type with empty name component");
      Scope scope0 = new Scope(node0, noType0);
      Scope.Var scope_Var0 = scope0.declare("x", node0, noType0, (CompilerInput) null, false);
      Node node2 = scope_Var0.getInitialValue();
      assertFalse(scope_Var0.isDefine());
      assertNotNull(node2);
      assertFalse(node2.hasChildren());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("oLOz7~", (Node) null, (JSType) null, (CompilerInput) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      scope_Var0.resolveType(simpleErrorReporter0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Node node0 = Node.newString("com.google.javascript.jscomp.Scope", 4095, 2072);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope scope0 = new Scope(node0, objectType0);
      Scope.Var scope_Var0 = scope0.declare("com.google.javascript.jscomp.WarningsGuard", node0, objectType0, (CompilerInput) null);
      scope_Var0.resolveType(simpleErrorReporter0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("com.google.javascript.jscomp.Scope");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0, false);
      Scope.Var scope_Var0 = scope0.declare("com.google.javascript.jscomp.Scope", (Node) null, (JSType) null, compilerInput0);
      String string0 = scope_Var0.getInputName();
      assertFalse(scope_Var0.isDefine());
      assertEquals("com.google.javascript.jscomp.Scope", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("Hk", (Node) null, (JSType) null, (CompilerInput) null);
      String string0 = scope_Var0.getInputName();
      assertFalse(scope_Var0.isDefine());
      assertEquals("<non-file>", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("Y<|NGI!vB+X", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isNoShadow();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Node node0 = Node.newString(4017, "`tP", 4017, 4017);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      JSDocInfo jSDocInfo0 = new JSDocInfo();
      node0.setJSDocInfo(jSDocInfo0);
      Scope.Var scope_Var0 = scope0.declare("com.google.javascript.jscomp.Scope$1", node0, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isNoShadow();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Node node0 = Node.newString("LEAST_FUNCTION_TYPE", (-418), (-3582));
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      NoType noType0 = (NoType)jSTypeRegistry0.getGreatestSubtypeWithProperty((JSType) null, "Named type with empty name component");
      Scope scope0 = new Scope(node0, noType0);
      Scope.Var scope_Var0 = scope0.declare("x", node0, noType0, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.equals(scope_Var0);
      assertFalse(scope_Var0.isDefine());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Node node0 = Node.newString("LEAST_FUNCTION_TYPE", (-418), (-3582));
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      NoType noType0 = (NoType)jSTypeRegistry0.getGreatestSubtypeWithProperty((JSType) null, "Named type with empty name component");
      Scope scope0 = new Scope(node0, noType0);
      Scope.Var scope_Var0 = scope0.declare("x", node0, noType0, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.equals((Object) null);
      assertFalse(scope_Var0.isDefine());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Node node0 = Node.newString("LEAST_FUNCTION_TYPE", (-418), (-3582));
      Node node1 = new Node(86, node0, node0, node0);
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      NoType noType0 = (NoType)jSTypeRegistry0.getGreatestSubtypeWithProperty((JSType) null, "Named type with empty name component");
      Scope scope0 = new Scope(node0, noType0);
      Scope.Var scope_Var0 = scope0.declare("x", node0, noType0, (CompilerInput) null, false);
      Scope.Var scope_Var1 = scope0.declare("_hq_,EEf>:j0Y,X", node1, noType0, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.equals(scope_Var1);
      assertFalse(scope_Var1.isDefine());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope scope1 = null;
      try {
        scope1 = new Scope(scope0, (Node) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newNumber(1288.227592706);
      Scope scope1 = new Scope(scope0, node0);
      Scope scope2 = scope1.getGlobalScope();
      assertTrue(scope2.isGlobal());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      // Undeclared exception!
      try { 
        scope0.declare((String) null, (Node) null, (JSType) null, (CompilerInput) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      // Undeclared exception!
      try { 
        scope0.declare("", (Node) null, (JSType) null, (CompilerInput) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      scope0.declare("com.google.javascript.jscomp.Scope$1", (Node) null, (JSType) null, (CompilerInput) null);
      // Undeclared exception!
      try { 
        scope0.declare("com.google.javascript.jscomp.Scope$1", (Node) null, (JSType) null, (CompilerInput) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("Hk", (Node) null, (JSType) null, (CompilerInput) null);
      Scope scope1 = new Scope((Node) null, (ObjectType) null);
      // Undeclared exception!
      try { 
        scope1.undeclare(scope_Var0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Node node0 = Node.newString(4033, "`tP", 4033, 4033);
      Scope scope0 = new Scope(node0, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("FZonRqBCGV'[", node0, (JSType) null, (CompilerInput) null);
      scope0.undeclare(scope_Var0);
      // Undeclared exception!
      try { 
        scope0.undeclare(scope_Var0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      scope0.declare("lw", (Node) null, (JSType) null, (CompilerInput) null, false);
      Scope.Var scope_Var0 = scope0.getVar("lw");
      assertFalse(scope_Var0.isDefine());
      assertNotNull(scope_Var0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newNumber((-1.0));
      Scope scope1 = new Scope(scope0, node0);
      Scope.Var scope_Var0 = scope1.getVar("com.google.common.base.CharMatcher$11");
      assertNull(scope_Var0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("PL6k:@qX;", (Node) null, (JSType) null, (CompilerInput) null);
      assertFalse(scope_Var0.isDefine());
      
      boolean boolean0 = scope0.isDeclared("PL6k:@qX;", false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newNumber((-7.7045636046287935));
      Scope scope1 = new Scope(scope0, node0);
      boolean boolean0 = scope1.isDeclared("5oWS+/kqDQa>:Z", true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newNumber(0.0);
      Scope scope1 = new Scope(scope0, node0);
      boolean boolean0 = scope1.isDeclared("5[@Ux?\"yj6> :", false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newNumber((-719.6862048332));
      Scope scope1 = new Scope(scope0, node0);
      boolean boolean0 = scope1.isLocal();
      assertTrue(boolean0);
  }
}