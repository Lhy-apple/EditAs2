/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:12:30 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.parser.Tag;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Tag_ESTest extends Tag_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Tag tag0 = Tag.valueOf("STYLE");
      String string0 = tag0.getName();
      assertEquals("style", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tag tag0 = Tag.valueOf("et");
      boolean boolean0 = tag0.canContainBlock();
      assertFalse(tag0.isData());
      assertTrue(boolean0);
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertFalse(tag0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("INPUT");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tag tag0 = Tag.valueOf("<jbTJDl_py");
      boolean boolean0 = tag0.preserveWhitespace();
      assertFalse(boolean0);
      assertFalse(tag0.isEmpty());
      assertFalse(tag0.isData());
      assertTrue(tag0.canContainBlock());
      assertTrue(tag0.isInline());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tag tag0 = Tag.valueOf("8:bd:d},6o");
      tag0.toString();
      assertTrue(tag0.isInline());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertFalse(tag0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tag tag0 = Tag.valueOf("body");
      boolean boolean0 = tag0.isBlock();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Tag tag0 = Tag.valueOf("dt");
      boolean boolean0 = tag0.canContain(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("head");
      boolean boolean0 = tag0.canContain(tag0);
      assertFalse(tag0.isData());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      boolean boolean0 = tag0.canContain(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Tag tag0 = Tag.valueOf("INPUT");
      boolean boolean0 = tag0.canContain(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Tag tag0 = Tag.valueOf("dt");
      Tag tag1 = Tag.valueOf("dd");
      boolean boolean0 = tag0.canContain(tag1);
      assertFalse(tag1.isData());
      assertFalse(tag1.equals((Object)tag0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tag tag0 = Tag.valueOf("head");
      Tag tag1 = Tag.valueOf("base");
      boolean boolean0 = tag0.canContain(tag1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tag tag0 = Tag.valueOf("script");
      Tag tag1 = tag0.getImplicitParent();
      assertNotNull(tag1);
      
      boolean boolean0 = tag1.canContain(tag0);
      assertTrue(boolean0);
      assertEquals("head", tag1.toString());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tag tag0 = Tag.valueOf("head");
      Tag tag1 = Tag.valueOf("noscript");
      boolean boolean0 = tag0.canContain(tag1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tag tag0 = Tag.valueOf("head");
      Tag tag1 = Tag.valueOf("link");
      boolean boolean0 = tag0.canContain(tag1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      Tag tag1 = tag0.getImplicitParent();
      boolean boolean0 = tag1.canContain(tag0);
      assertTrue(boolean0);
      assertEquals("head", tag1.toString());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Tag tag0 = Tag.valueOf("TITLE");
      Tag tag1 = tag0.getImplicitParent();
      boolean boolean0 = tag1.canContain(tag0);
      assertTrue(boolean0);
      assertEquals("head", tag1.toString());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Tag tag0 = Tag.valueOf("STYLE");
      Tag tag1 = tag0.getImplicitParent();
      boolean boolean0 = tag1.canContain(tag0);
      assertEquals("head", tag1.toString());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Tag tag0 = Tag.valueOf("OBJECT");
      Tag tag1 = tag0.getImplicitParent();
      boolean boolean0 = tag1.canContain(tag0);
      assertTrue(boolean0);
      assertEquals("head", tag1.getName());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Tag tag0 = Tag.valueOf("dt");
      Tag tag1 = Tag.valueOf("dd");
      tag1.canContain(tag0);
      assertFalse(tag1.equals((Object)tag0));
      assertFalse(tag1.isData());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Tag tag0 = Tag.valueOf("dt");
      Tag tag1 = Tag.valueOf("et");
      boolean boolean0 = tag0.canContain(tag1);
      assertTrue(tag1.canContainBlock());
      assertTrue(boolean0);
      assertFalse(tag1.isData());
      assertFalse(tag1.isEmpty());
      assertFalse(tag1.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Tag tag0 = Tag.valueOf("dd");
      Tag tag1 = Tag.valueOf("e/");
      boolean boolean0 = tag0.canContain(tag1);
      assertFalse(tag1.isData());
      assertFalse(tag1.preserveWhitespace());
      assertTrue(boolean0);
      assertFalse(tag1.isEmpty());
      assertTrue(tag1.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Tag tag0 = Tag.valueOf("head");
      boolean boolean0 = tag0.isInline();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("heF6d");
      boolean boolean0 = tag0.isInline();
      assertTrue(boolean0);
      assertFalse(tag0.isEmpty());
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Tag tag0 = Tag.valueOf("SCRIPT");
      boolean boolean0 = tag0.isData();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Tag tag0 = Tag.valueOf("html");
      Tag tag1 = tag0.getImplicitParent();
      assertNull(tag1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Tag tag0 = Tag.valueOf("head");
      boolean boolean0 = tag0.isValidParent(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Tag tag0 = Tag.valueOf("head");
      Tag tag1 = tag0.getImplicitParent();
      boolean boolean0 = tag0.isValidParent(tag1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Tag tag0 = Tag.valueOf("SMALL");
      Tag tag1 = tag0.getImplicitParent();
      boolean boolean0 = tag1.isValidParent(tag0);
      assertFalse(tag1.equals((Object)tag0));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Tag tag0 = Tag.valueOf("script");
      boolean boolean0 = tag0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Tag tag0 = Tag.valueOf("The validated array contains null element at index: ");
      boolean boolean0 = tag0.equals("The validated array contains null element at index: ");
      assertFalse(tag0.isEmpty());
      assertFalse(boolean0);
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Tag tag0 = Tag.valueOf("dd");
      Tag tag1 = Tag.valueOf("PARAM");
      boolean boolean0 = tag0.canContain(tag1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Tag tag0 = Tag.valueOf("bt");
      boolean boolean0 = tag0.isValidParent(tag0);
      assertFalse(boolean0);
      assertFalse(tag0.isEmpty());
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Tag tag0 = Tag.valueOf("TBODY");
      Tag tag1 = tag0.getImplicitParent();
      boolean boolean0 = tag0.canContain(tag1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Tag tag0 = Tag.valueOf("hek");
      Tag tag1 = Tag.valueOf("hek");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(tag1.preserveWhitespace());
      assertTrue(tag1.canContainBlock());
      assertTrue(boolean0);
      assertTrue(tag1.isInline());
      assertFalse(tag1.isData());
      assertFalse(tag1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Tag tag0 = Tag.valueOf("head");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Tag tag0 = Tag.valueOf("dt");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Tag tag0 = Tag.valueOf("link");
      tag0.hashCode();
  }
}