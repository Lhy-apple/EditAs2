/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:07:03 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerPass;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.MakeDeclaredNamesUnique;
import com.google.javascript.jscomp.MessageFormatter;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.rhino.Node;
import java.util.NoSuchElementException;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MakeDeclaredNamesUnique_ESTest extends MakeDeclaredNamesUnique_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerPass compilerPass0 = MakeDeclaredNamesUnique.getContextualRenameInverter(compiler0);
      Node node0 = compiler0.parseTestCode("nvut|6&R");
      compilerPass0.process(node0, node0);
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("K7_|LJ,add");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, makeDeclaredNamesUnique0);
      nodeTraversal0.traverse(node0);
      assertEquals(41, Node.BRACELESS_TYPE);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, "nvut|6&R", false);
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique(makeDeclaredNamesUnique_InlineRenamer0);
      Node node0 = compiler0.parseTestCode("nvut|6&R");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, makeDeclaredNamesUnique0);
      nodeTraversal0.traverse(node0);
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      boolean boolean0 = makeDeclaredNamesUnique_ContextualRenamer0.stripConstIfReplaced();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      makeDeclaredNamesUnique_ContextualRenamer0.addDeclaredName("");
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer1 = (MakeDeclaredNamesUnique.ContextualRenamer)makeDeclaredNamesUnique_ContextualRenamer0.forChildScope();
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("");
      assertFalse(makeDeclaredNamesUnique_ContextualRenamer0.equals((Object)makeDeclaredNamesUnique_ContextualRenamer1));
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("r9=jhw,");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, makeDeclaredNamesUnique0);
      NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) makeDeclaredNamesUnique0);
      nodeTraversal0.traverse(node0);
      assertFalse(node0.hasMoreThanOneChild());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer0 = new MakeDeclaredNamesUnique.ContextualRenamer();
      MakeDeclaredNamesUnique.ContextualRenamer makeDeclaredNamesUnique_ContextualRenamer1 = (MakeDeclaredNamesUnique.ContextualRenamer)makeDeclaredNamesUnique_ContextualRenamer0.forChildScope();
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("");
      makeDeclaredNamesUnique_ContextualRenamer1.addDeclaredName("");
      assertFalse(makeDeclaredNamesUnique_ContextualRenamer1.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Logger logger0 = Logger.getLogger("O{&5 ;#{5");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager((MessageFormatter) null, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, "O{&5 ;#{5", false);
      boolean boolean0 = makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, "nvut|6&R", false);
      makeDeclaredNamesUnique_InlineRenamer0.forChildScope();
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      String string0 = MakeDeclaredNamesUnique.ContextualRenameInverter.getOrginalName("X%cEuqiR9OD");
      assertEquals("X%cEuqiR9OD", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      Compiler compiler0 = new Compiler();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, makeDeclaredNamesUnique0);
      Node node0 = compiler0.parseTestCode("IN: %s OUT: %s");
      Node node1 = new Node(105, node0, node0, node0);
      // Undeclared exception!
      try { 
        makeDeclaredNamesUnique0.visit(nodeTraversal0, node1, node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayDeque", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      String string0 = MakeDeclaredNamesUnique.ContextualRenameInverter.getOrginalName("$$");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = null;
      try {
        makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, "", true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, "nvut|6&R", false);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("");
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, "d~}2^w7`c!fpU*^o", false);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("~O");
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("~O");
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      MakeDeclaredNamesUnique.InlineRenamer makeDeclaredNamesUnique_InlineRenamer0 = new MakeDeclaredNamesUnique.InlineRenamer(supplier0, "X%cEuqiR9OD", false);
      makeDeclaredNamesUnique_InlineRenamer0.addDeclaredName("$$");
      assertFalse(makeDeclaredNamesUnique_InlineRenamer0.stripConstIfReplaced());
  }
}