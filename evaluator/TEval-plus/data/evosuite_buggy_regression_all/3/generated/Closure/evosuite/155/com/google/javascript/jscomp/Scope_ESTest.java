/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:23:33 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.LinkedFlowScope;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.StaticSlot;
import java.util.List;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Scope_ESTest extends Scope_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("IJP]\"sDn-+,{`EMr", "IJP]\"sDn-+,{`EMr");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope.Var scope_Var0 = scope0.declare("IJP]\"sDn-+,{`EMr", node0, (JSType) null, (CompilerInput) null, true);
      boolean boolean0 = scope_Var0.isLocal();
      assertFalse(scope_Var0.isDefine());
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.isGlobal();
      assertFalse(scope_Var0.isDefine());
      assertTrue(boolean0);
      assertFalse(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, false);
      scope_Var0.getJSDocInfo();
      assertFalse(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("IJP]\"sDn-+,{`EMr", "IJP]\"sDn-+,{`EMr");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope.Var scope_Var0 = scope0.declare("IJP]\"sDn-+,{`EMr", node0, (JSType) null, (CompilerInput) null, true);
      String string0 = scope_Var0.getName();
      assertNotNull(string0);
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, true);
      Node node1 = scope_Var0.getNameNode();
      assertFalse(scope_Var0.isDefine());
      assertTrue(scope_Var0.isTypeInferred());
      assertNotNull(node1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "<non-file>");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("<non-file>", (Node) null, (JSType) null, (CompilerInput) null, false);
      scope_Var0.getScope();
      assertFalse(scope_Var0.isDefine());
      assertFalse(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, true);
      scope_Var0.getType();
      assertFalse(scope_Var0.isDefine());
      assertTrue(scope_Var0.isTypeInferred());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("TTpG y9?C`O,Py ", "com.google.javascript.jscomp.Scope$Var");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("com.google.javascript.jscomp.Scope$Var", node0, (JSType) null, (CompilerInput) null);
      assertTrue(scope_Var0.isTypeInferred());
      
      scope_Var0.setType((JSType) null);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("IJP]\"sDn-+,{`EMr", "IJP]\"sDn-+,{`EMr");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope.Var scope_Var0 = scope0.declare("IJP]\"sDn-+,{`EMr", node0, (JSType) null, (CompilerInput) null, true);
      boolean boolean0 = scope_Var0.isDefine();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null);
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
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, true);
      String string0 = scope_Var0.toString();
      assertEquals("Scope.Var 84uGN49A#", string0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      JSTypeNative jSTypeNative0 = JSTypeNative.NO_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      Scope scope0 = new Scope(node0, functionType0);
      assertFalse(scope0.isLocal());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Scope scope0 = new Scope((Node) null, compiler0);
      boolean boolean0 = scope0.isBottom();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uG9A#", "84uG9A#");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createScope(node0, (Scope) null);
      StaticSlot<JSType> staticSlot0 = scope0.getSlot("// Input %num%");
      assertNull(staticSlot0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Node node1 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      Scope.Var scope_Var0 = scope1.getVar("84uGN49A#");
      assertNull(scope_Var0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      int int0 = scope0.getDepth();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Scope scope0 = new Scope((Node) null, compiler0);
      int int0 = scope0.getVarCount();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Scope scope0 = new Scope((Node) null, compiler0);
      LinkedFlowScope linkedFlowScope0 = LinkedFlowScope.createEntryLattice(scope0);
      JSType jSType0 = linkedFlowScope0.getTypeOfThis();
      assertFalse(jSType0.isNoObjectType());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84dGN49u#", "84dGN49u#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      StaticSlot<JSType> staticSlot0 = scope0.getOwnSlot("com.google.javascript.jscomp.Scope$1");
      assertNull(staticSlot0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("txVw/Kq<46>{", (Node) null, (JSType) null, (CompilerInput) null, false);
      // Undeclared exception!
      try { 
        scope_Var0.getInitialValue();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Scope$Var", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "<non-file>");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("<non-file>", (Node) null, (JSType) null, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.isExtern();
      assertTrue(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("TTpG y9?C`O,Py ", "com.google.javascript.jscomp.Scope$Var");
      Scope scope0 = new Scope(node0, compiler0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("TTpG y9?C`O,Py ", "TTpG y9?C`O,Py ");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      Scope.Var scope_Var0 = scope0.declare("0:fqhDi2H}r9q c{p", node0, (JSType) null, compilerInput0, true);
      boolean boolean0 = scope_Var0.isExtern();
      assertFalse(scope_Var0.isDefine());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Scope scope0 = new Scope((Node) null, compiler0);
      Scope.Var scope_Var0 = scope0.declare("cJ_(O$Y/.G#*", (Node) null, (JSType) null, (CompilerInput) null, true);
      boolean boolean0 = scope_Var0.isConst();
      assertFalse(scope_Var0.isDefine());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isConst();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Node node1 = new Node(1, node0, 1, 27);
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, false);
      Node node2 = scope_Var0.getInitialValue();
      assertNull(node2);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "<non-file>");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("<non-file>", node0, (JSType) null, (CompilerInput) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      scope_Var0.resolveType(simpleErrorReporter0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Scope scope0 = new Scope((Node) null, compiler0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Vector<JSType> vector0 = new Vector<JSType>();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionTypeWithVarArgs((JSType) null, (List<JSType>) vector0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("Not declared as a constructor", "c;_Lj)m&MC\"{*i^");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      Scope.Var scope_Var0 = scope0.declare("com.google.javascript.jscomp.MakeDeclaredNamesUnique", (Node) null, functionType0, compilerInput0, true);
      String string0 = scope_Var0.getInputName();
      assertFalse(scope_Var0.isDefine());
      assertEquals("Not declared as a constructor", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, true);
      String string0 = scope_Var0.getInputName();
      assertEquals("<non-file>", string0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Scope scope0 = new Scope((Node) null, compiler0);
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      Vector<JSType> vector0 = new Vector<JSType>();
      FunctionType functionType0 = jSTypeRegistry0.createFunctionTypeWithVarArgs((JSType) null, (List<JSType>) vector0);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("Not declared as a constructor", "c;_Lj)m&MC\"{*i^");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      Scope.Var scope_Var0 = scope0.declare("com.google.javascript.jscomp.MakeDeclaredNamesUnique", (Node) null, functionType0, compilerInput0, true);
      boolean boolean0 = scope_Var0.isNoShadow();
      assertEquals("Not declared as a constructor", scope_Var0.getInputName());
      assertFalse(scope_Var0.isDefine());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      node0.addSuppression("84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("txVw/Kq<46>{", node0, (JSType) null, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.isNoShadow();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "<non-file>");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("<non-file>", (Node) null, (JSType) null, (CompilerInput) null, false);
      boolean boolean0 = scope_Var0.equals(scope_Var0);
      assertTrue(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.equals(scope0);
      assertFalse(scope_Var0.isDefine());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Scope scope0 = new Scope((Node) null, compiler0);
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
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "<non-file>");
      Scope scope0 = new Scope(node0, compiler0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node1 = compiler0.parseTestCode("84uGN49A#");
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      Scope scope2 = scope1.getGlobalScope();
      assertTrue(scope2.isGlobal());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      // Undeclared exception!
      try { 
        scope0.declare((String) null, node0, (JSType) null, (CompilerInput) null, false);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getErrorManager();
      Scope scope0 = new Scope((Node) null, compiler0);
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
  public void test36()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, false);
      // Undeclared exception!
      try { 
        scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Node node1 = new Node(1, node0, 1, 27);
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, true);
      Scope scope1 = new Scope(scope0, node1);
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
  public void test38()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null, true);
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
  public void test39()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "<non-file>");
      Scope scope0 = new Scope(node0, compiler0);
      scope0.declare("<non-file>", (Node) null, (JSType) null, (CompilerInput) null, false);
      Scope.Var scope_Var0 = scope0.getVar("<non-file>");
      assertNotNull(scope_Var0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Scope scope0 = new Scope(node0, compiler0);
      Scope.Var scope_Var0 = scope0.declare("84uGN49A#", node0, (JSType) null, (CompilerInput) null);
      assertFalse(scope_Var0.isDefine());
      
      boolean boolean0 = scope0.isDeclared("84uGN49A#", false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Node node1 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      boolean boolean0 = scope1.isDeclared("84uGN49A#", true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Node node1 = new Node(1, node0, 1, 27);
      Scope scope0 = new Scope(node1, compiler0);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      boolean boolean0 = scope1.isDeclared("84uGN49A#", false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      Node node1 = compiler0.parseSyntheticCode("84uGN49A#", "84uGN49A#");
      ClosureCodingConvention closureCodingConvention0 = (ClosureCodingConvention)compiler0.defaultCodingConvention;
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Scope scope1 = typedScopeCreator0.createScope(node1, scope0);
      boolean boolean0 = scope1.isLocal();
      assertTrue(boolean0);
  }
}
