/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:49:40 GMT 2023
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
      Tag tag0 = Tag.valueOf("hr");
      boolean boolean0 = tag0.isData();
      assertTrue(tag0.isEmpty());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br]\"2F>t}u]lz");
      assertNotNull(tag0);
      
      boolean boolean0 = tag0.preserveWhitespace();
      assertFalse(tag0.isSelfClosing());
      assertFalse(boolean0);
      assertTrue(tag0.isInline());
      assertFalse(tag0.isData());
      assertTrue(tag0.canContainBlock());
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("pre");
      boolean boolean0 = tag0.formatAsBlock();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tag tag0 = Tag.valueOf("div");
      String string0 = tag0.getName();
      assertEquals("div", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tag tag0 = Tag.valueOf("div");
      String string0 = tag0.toString();
      assertEquals("div", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tag tag0 = Tag.valueOf("atji");
      boolean boolean0 = tag0.isBlock();
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertFalse(boolean0);
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Tag tag0 = Tag.valueOf("String must not be empty");
      boolean boolean0 = tag0.canContainBlock();
      assertTrue(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("hr");
      boolean boolean0 = tag0.isInline();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tag tag0 = Tag.valueOf("{WXaUOt");
      boolean boolean0 = tag0.isInline();
      assertTrue(tag0.formatAsBlock());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Tag tag0 = Tag.valueOf("String must not be empty");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.canContainBlock());
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Tag tag0 = Tag.valueOf("hr");
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tag tag0 = Tag.valueOf("tt");
      boolean boolean0 = tag0.isSelfClosing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br]\"2F>t}u]lz");
      boolean boolean0 = tag0.isSelfClosing();
      assertFalse(boolean0);
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertTrue(tag0.formatAsBlock());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tag tag0 = Tag.valueOf("String must not be empty");
      boolean boolean0 = tag0.isKnownTag();
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertFalse(boolean0);
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.isInline());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tag tag0 = Tag.valueOf("hr");
      boolean boolean0 = tag0.isKnownTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("Atji");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("hr");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Tag tag0 = Tag.valueOf("String must not be empty");
      Tag tag1 = Tag.valueOf("pre");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isData());
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Tag tag0 = Tag.valueOf("org.jsoup.helper.Validate");
      boolean boolean0 = tag0.equals(tag0);
      assertFalse(tag0.isData());
      assertTrue(boolean0);
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Tag tag0 = Tag.valueOf("String must not be empty");
      Object object0 = new Object();
      boolean boolean0 = tag0.equals(object0);
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(boolean0);
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Tag tag0 = Tag.valueOf("hr");
      Tag tag1 = Tag.valueOf("}");
      boolean boolean0 = tag0.equals(tag1);
      assertTrue(tag1.canContainBlock());
      assertFalse(boolean0);
      assertFalse(tag1.isData());
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isSelfClosing());
      assertTrue(tag1.isInline());
      assertFalse(tag1.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Tag tag0 = Tag.valueOf("command");
      Tag tag1 = Tag.valueOf("datalist");
      assertFalse(tag1.isSelfClosing());
      
      Tag tag2 = tag1.setSelfClosing();
      boolean boolean0 = tag0.equals(tag2);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Tag tag0 = Tag.valueOf("pre");
      Tag tag1 = Tag.valueOf("h3");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("nav");
      Tag tag1 = Tag.valueOf("M*ust be true");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(tag1.isData());
      assertTrue(tag1.canContainBlock());
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isSelfClosing());
      assertTrue(tag1.isInline());
      assertFalse(tag1.preserveWhitespace());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Tag tag0 = Tag.valueOf("ZNfzwYFu]7m+ L)FA");
      Tag tag1 = Tag.valueOf("ZNfzwYVu]7m+ L)FA");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.isBlock());
      assertFalse(tag1.isData());
      assertTrue(tag1.canContainBlock());
      assertFalse(tag1.preserveWhitespace());
      assertFalse(boolean0);
      assertTrue(tag1.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Tag tag0 = Tag.valueOf("org.jsoup.helper.Validate");
      Tag tag1 = Tag.valueOf("org.jsoup.helper.Validate");
      assertFalse(tag1.isSelfClosing());
      
      tag1.setSelfClosing();
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Tag tag0 = Tag.valueOf("org.jsoup.helper.Validate");
      Tag tag1 = Tag.valueOf("org.jsoup.helper.Validate");
      boolean boolean0 = tag1.equals(tag0);
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isData());
      assertTrue(boolean0);
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.preserveWhitespace());
      assertTrue(tag1.isInline());
      assertTrue(tag1.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Tag tag0 = Tag.valueOf("pre");
      tag0.hashCode();
  }
}