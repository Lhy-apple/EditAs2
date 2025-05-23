/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:32:06 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Predicate;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.DefaultCodingConvention;
import com.google.javascript.jscomp.GoogleCodingConvention;
import com.google.javascript.jscomp.InferJSDocInfo;
import com.google.javascript.jscomp.MemoizedScopeCreator;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.ReferenceCollectingCallback;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ReferenceCollectingCallback_ESTest extends ReferenceCollectingCallback_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, (Predicate<Scope.Var>) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0, typedScopeCreator0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_Reference0.getAssignedValue();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, referenceCollectingCallback_BasicBlock0);
      Scope scope0 = referenceCollectingCallback_Reference0.getScope();
      assertNull(scope0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, referenceCollectingCallback_BasicBlock0);
      Node node1 = referenceCollectingCallback_Reference0.getParent();
      assertEquals(1, Node.FLAG_GLOBAL_STATE_UNMODIFIED);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, (Predicate<Scope.Var>) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0, typedScopeCreator0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, referenceCollectingCallback_BasicBlock0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock1 = referenceCollectingCallback_Reference0.getBasicBlock();
      assertSame(referenceCollectingCallback_BasicBlock0, referenceCollectingCallback_BasicBlock1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, referenceCollectingCallback_BasicBlock0);
      String string0 = referenceCollectingCallback_Reference0.getSourceName();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      Normalize.NormalizeStatements normalize_NormalizeStatements0 = new Normalize.NormalizeStatements(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, normalize_NormalizeStatements0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, referenceCollectingCallback_BasicBlock0);
      Node node1 = referenceCollectingCallback_Reference0.getGrandparent();
      assertNull(node1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, (Predicate<Scope.Var>) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0, typedScopeCreator0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_Reference0.isHoistedFunction();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      InferJSDocInfo inferJSDocInfo0 = new InferJSDocInfo((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, inferJSDocInfo0, (ScopeCreator) null);
      referenceCollectingCallback_ReferenceCollection0.add((ReferenceCollectingCallback.Reference) null, nodeTraversal0, (Scope.Var) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isWellDefined();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      Predicate<Scope.Var> predicate0 = (Predicate<Scope.Var>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, predicate0);
      Set<Scope.Var> set0 = referenceCollectingCallback0.getReferencedVariables();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", "com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection");
      referenceCollectingCallback0.process(node0, node0);
      assertEquals(28, Node.DEBUGSOURCE_PROP);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = referenceCollectingCallback0.getReferenceCollection((Scope.Var) null);
      assertNull(referenceCollectingCallback_ReferenceCollection0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock1 = new ReferenceCollectingCallback.BasicBlock(referenceCollectingCallback_BasicBlock0, node0);
      boolean boolean0 = referenceCollectingCallback_BasicBlock1.provablyExecutesBefore(referenceCollectingCallback_BasicBlock0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(100, 1, (-308));
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0, (ScopeCreator) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(77, 1, (-281));
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      boolean boolean0 = referenceCollectingCallback0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("?k", "?k");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, (ScopeCreator) null);
      Node node1 = new Node(113, 1, (-308));
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      Predicate<Scope.Var> predicate0 = (Predicate<Scope.Var>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, predicate0);
      boolean boolean0 = referenceCollectingCallback0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("?k", ">;oVq3U>`kA[]d");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, (ScopeCreator) null);
      Node node1 = new Node(119, 42, 32);
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      Predicate<Scope.Var> predicate0 = (Predicate<Scope.Var>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, predicate0);
      boolean boolean0 = referenceCollectingCallback0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("?k", "?k");
      GoogleCodingConvention googleCodingConvention0 = new GoogleCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, googleCodingConvention0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      Node node1 = new Node(111, node0, node0);
      Predicate<Scope.Var> predicate0 = (Predicate<Scope.Var>) mock(Predicate.class, new ViolatedAssumptionAnswer());
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, (ReferenceCollectingCallback.Behavior) null, predicate0);
      boolean boolean0 = referenceCollectingCallback0.shouldTraverse(nodeTraversal0, node1, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isWellDefined();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isEscaped();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      LinkedList<ReferenceCollectingCallback.Reference> linkedList0 = new LinkedList<ReferenceCollectingCallback.Reference>();
      referenceCollectingCallback_ReferenceCollection0.references = (List<ReferenceCollectingCallback.Reference>) linkedList0;
      linkedList0.addLast((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isEscaped();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$Reference", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = referenceCollectingCallback_ReferenceCollection0.getInitializingReferenceForConstants();
      assertNull(referenceCollectingCallback_Reference0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      LinkedList<ReferenceCollectingCallback.Reference> linkedList0 = new LinkedList<ReferenceCollectingCallback.Reference>();
      referenceCollectingCallback_ReferenceCollection0.references = (List<ReferenceCollectingCallback.Reference>) linkedList0;
      linkedList0.addLast((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.getInitializingReferenceForConstants();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isAssignedOnceInLifetime();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      LinkedList<ReferenceCollectingCallback.Reference> linkedList0 = new LinkedList<ReferenceCollectingCallback.Reference>();
      referenceCollectingCallback_ReferenceCollection0.references = (List<ReferenceCollectingCallback.Reference>) linkedList0;
      linkedList0.addLast((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.isAssignedOnceInLifetime();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.isNeverAssigned();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      boolean boolean0 = referenceCollectingCallback_ReferenceCollection0.firstReferenceIsAssigningDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ReferenceCollectingCallback.ReferenceCollection referenceCollectingCallback_ReferenceCollection0 = new ReferenceCollectingCallback.ReferenceCollection();
      LinkedList<ReferenceCollectingCallback.Reference> linkedList0 = new LinkedList<ReferenceCollectingCallback.Reference>();
      referenceCollectingCallback_ReferenceCollection0.references = (List<ReferenceCollectingCallback.Reference>) linkedList0;
      linkedList0.addLast((ReferenceCollectingCallback.Reference) null);
      // Undeclared exception!
      try { 
        referenceCollectingCallback_ReferenceCollection0.firstReferenceIsAssigningDeclaration();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.ReferenceCollectingCallback$ReferenceCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0, (Predicate<Scope.Var>) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0, typedScopeCreator0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, referenceCollectingCallback_BasicBlock0);
      boolean boolean0 = referenceCollectingCallback_Reference0.isDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("&", "&");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isVarDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("&", "&");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isInitializingDeclaration();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("&", "&");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      node0.addChildToBack(node0);
      boolean boolean0 = referenceCollectingCallback_Reference0.isLvalue();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("&", "&");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = new ReferenceCollectingCallback.Reference(node0, node0, nodeTraversal0, (ReferenceCollectingCallback.BasicBlock) null);
      boolean boolean0 = referenceCollectingCallback_Reference0.isSimpleAssignmentToName();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("?k", "?k");
      Node node1 = new Node(86, node0, node0);
      DefaultCodingConvention defaultCodingConvention0 = new DefaultCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, defaultCodingConvention0);
      MemoizedScopeCreator memoizedScopeCreator0 = new MemoizedScopeCreator(typedScopeCreator0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, memoizedScopeCreator0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, referenceCollectingCallback_BasicBlock0, node1);
      boolean boolean0 = referenceCollectingCallback_Reference0.isSimpleAssignmentToName();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("?k", "?k");
      Node node1 = new Node(86, node0, node0);
      DefaultCodingConvention defaultCodingConvention0 = new DefaultCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, defaultCodingConvention0);
      MemoizedScopeCreator memoizedScopeCreator0 = new MemoizedScopeCreator(typedScopeCreator0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, memoizedScopeCreator0);
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      ReferenceCollectingCallback.Reference referenceCollectingCallback_Reference0 = ReferenceCollectingCallback.Reference.newBleedingFunction(nodeTraversal0, referenceCollectingCallback_BasicBlock0, node1);
      boolean boolean0 = referenceCollectingCallback_Reference0.isLvalue();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode(">", ">");
      ReferenceCollectingCallback.BasicBlock referenceCollectingCallback_BasicBlock0 = new ReferenceCollectingCallback.BasicBlock((ReferenceCollectingCallback.BasicBlock) null, node0);
      boolean boolean0 = referenceCollectingCallback_BasicBlock0.provablyExecutesBefore(referenceCollectingCallback_BasicBlock0);
      assertTrue(boolean0);
  }
}
