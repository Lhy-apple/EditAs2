/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:41:07 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import com.google.javascript.rhino.jstype.ObjectType;
import com.google.javascript.rhino.jstype.StaticScope;
import com.google.javascript.rhino.jstype.StaticSlot;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Scope_ESTest extends Scope_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("BY_PART", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isLocal();
      assertFalse(scope_Var0.isDefine());
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("com.google.javascript.jscomp.Scope$Var");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0, false);
      Scope.Var scope_Var0 = scope0.declare("ii8]haq1n{,", (Node) null, (JSType) null, compilerInput0, false);
      boolean boolean0 = scope_Var0.isGlobal();
      assertTrue(boolean0);
      assertFalse(scope_Var0.isDefine());
      assertFalse(scope_Var0.isTypeInferred());
      assertEquals("com.google.javascript.jscomp.Scope$Var", scope_Var0.getInputName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare(">MQ f@Qa{NLdQ0!V", (Node) null, (JSType) null, (CompilerInput) null);
      scope_Var0.getJSDocInfo();
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("7Bz_9V(w<>F*gaB");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0, false);
      Scope.Var scope_Var0 = scope0.declare("7Bz_9V(w<>F*gaB", (Node) null, (JSType) null, compilerInput0, true);
      String string0 = scope_Var0.getName();
      assertEquals("7Bz_9V(w<>F*gaB", scope_Var0.getInputName());
      assertNotNull(string0);
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("O~v45{:@gYb", (Node) null, (JSType) null, (CompilerInput) null);
      scope_Var0.getNameNode();
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("GX'~/vr9RJ\"-", (Node) null, (JSType) null, (CompilerInput) null);
      scope_Var0.getScope();
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("}kFwd)y{S", (Node) null, (JSType) null, (CompilerInput) null);
      scope_Var0.getType();
      assertTrue(scope_Var0.isTypeInferred());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("GX'~s/vr9RJ\"-", (Node) null, (JSType) null, (CompilerInput) null);
      assertTrue(scope_Var0.isTypeInferred());
      
      scope_Var0.setType((JSType) null);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare(">MQ f@Qa{NLdQ0!V", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isDefine();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("JSC_INVALID_TWEAK_DEFAULT_VALUE_WARNING", (Node) null, (JSType) null, (CompilerInput) null);
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
      Scope.Var scope_Var0 = scope0.declare("aq*9`)rW7?PkOOPkm", (Node) null, (JSType) null, (CompilerInput) null);
      String string0 = scope_Var0.toString();
      assertFalse(scope_Var0.isDefine());
      assertEquals("Scope.Var aq*9`)rW7?PkOOPkm", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Node node0 = Node.newString(786, "");
      // Undeclared exception!
      try { 
        syntacticScopeCreator0.createScope(node0, (Scope) null);
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
      StaticSlot<JSType> staticSlot0 = scope0.getSlot("right");
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
      ObjectType objectType0 = scope0.getTypeOfThis();
      assertNull(objectType0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      StaticSlot<JSType> staticSlot0 = scope0.getOwnSlot("");
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
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newNumber(0.0);
      Scope.Var scope_Var0 = scope0.declare(">MQ f@Qaj{NLdQ0!V", node0, (JSType) null, (CompilerInput) null);
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
  public void test22()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("}kFwd)y{S", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isExtern();
      assertFalse(scope_Var0.isDefine());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("FT&qSly8!LQQcHTP");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0, true);
      Scope.Var scope_Var0 = scope0.declare("FT&qSly8!LQQcHTP", (Node) null, (JSType) null, compilerInput0, true);
      boolean boolean0 = scope_Var0.isExtern();
      assertFalse(scope_Var0.isDefine());
      assertEquals("FT&qSly8!LQQcHTP", scope_Var0.getInputName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("com.google.javascript.jscomp.Scope$Var");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0, false);
      Scope.Var scope_Var0 = scope0.declare("ii8]haq1n{,", (Node) null, (JSType) null, compilerInput0, false);
      boolean boolean0 = scope_Var0.isExtern();
      assertFalse(scope_Var0.isDefine());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("BY_PART", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isConst();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newNumber(0.0);
      Scope.Var scope_Var0 = scope0.declare(">MQ f@Qaj{NLdQ0!V", node0, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.isConst();
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("GX'~s/vr9RJ\"-", (Node) null, (JSType) null, (CompilerInput) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      scope_Var0.resolveType(simpleErrorReporter0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      ObjectType objectType0 = jSTypeRegistry0.createAnonymousObjectType();
      Scope.Var scope_Var0 = scope0.declare("-$:a)5mY9oL^D*wt", (Node) null, objectType0, (CompilerInput) null);
      scope_Var0.resolveType(simpleErrorReporter0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("'ZW{-");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      Scope.Var scope_Var0 = scope0.declare("goog.isObject", (Node) null, (JSType) null, compilerInput0);
      String string0 = scope_Var0.getInputName();
      assertEquals("'ZW{-", string0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("NqA]O^L]F1UYw", (Node) null, (JSType) null, (CompilerInput) null);
      String string0 = scope_Var0.getInputName();
      assertEquals("<non-file>", string0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("'ZW{-");
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      Scope.Var scope_Var0 = scope0.declare("goog.isObject", (Node) null, (JSType) null, compilerInput0);
      boolean boolean0 = scope_Var0.isNoShadow();
      assertFalse(boolean0);
      assertEquals("'ZW{-", scope_Var0.getInputName());
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("Cy", (SourceFile.Generator) null);
      CompilerInput compilerInput0 = new CompilerInput(jSSourceFile0);
      Scope.Var scope_Var0 = scope0.declare("Cy", (Node) null, (JSType) null, compilerInput0);
      boolean boolean0 = scope_Var0.equals(scope_Var0);
      assertFalse(scope_Var0.isDefine());
      assertEquals("Cy", scope_Var0.getInputName());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("BY_PART", (Node) null, (JSType) null, (CompilerInput) null);
      boolean boolean0 = scope_Var0.equals((Object) null);
      assertFalse(boolean0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
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
  public void test35()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newString(187, "b)J7");
      Scope scope1 = new Scope(scope0, node0);
      Scope scope2 = scope1.getGlobalScope();
      assertTrue(scope2.isGlobal());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
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
  public void test37()  throws Throwable  {
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
  public void test38()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      scope0.declare("YR0-<]@>JQ=2Z,5 ", (Node) null, (JSType) null, (CompilerInput) null);
      // Undeclared exception!
      try { 
        scope0.declare("YR0-<]@>JQ=2Z,5 ", (Node) null, (JSType) null, (CompilerInput) null);
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
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("yBwA\"sf:p3L.]C+vQpk", (Node) null, (JSType) null, (CompilerInput) null, true);
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
  public void test40()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare("yBwA\"sf:p3L.]C+vQpk", (Node) null, (JSType) null, (CompilerInput) null, false);
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
  public void test41()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      scope0.declare(">MQ f@Qa{NLdQ0!V", (Node) null, (JSType) null, (CompilerInput) null);
      Scope.Var scope_Var0 = scope0.getVar(">MQ f@Qa{NLdQ0!V");
      assertNotNull(scope_Var0);
      assertFalse(scope_Var0.isDefine());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newString((-1468), "i?2RCG*oZ[8TC");
      Scope scope1 = new Scope(scope0, node0);
      Scope.Var scope_Var0 = scope1.getVar("com.google.common.base.Joiner$MapJoiner");
      assertNull(scope_Var0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Scope.Var scope_Var0 = scope0.declare(">MQ f@Qa{NLdQ0!V", (Node) null, (JSType) null, (CompilerInput) null);
      assertFalse(scope_Var0.isDefine());
      
      boolean boolean0 = scope0.isDeclared(">MQ f@Qa{NLdQ0!V", false);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      Scope scope1 = new Scope(scope0, node0);
      boolean boolean0 = scope1.isDeclared("JSC_INVALID_TWEAK_DEFAULT_VALUE_WARNING", false);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null, true);
      Stack<JSType> stack0 = new Stack<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) stack0);
      Scope scope1 = new Scope(scope0, node0);
      boolean boolean0 = scope1.isDeclared("com.google.javascript.jscomp.SanityCheck", true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Scope scope0 = new Scope((Node) null, (ObjectType) null);
      Node node0 = Node.newString((-663), ">MQ f@Qa{NLdQ0!V");
      Scope scope1 = new Scope(scope0, node0);
      boolean boolean0 = scope1.isLocal();
      assertTrue(boolean0);
  }
}