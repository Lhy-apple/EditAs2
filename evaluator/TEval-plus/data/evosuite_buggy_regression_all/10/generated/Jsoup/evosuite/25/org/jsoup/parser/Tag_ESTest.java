/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:50:32 GMT 2023
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
      Tag tag0 = Tag.valueOf("link");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
      assertTrue(tag0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tag tag0 = Tag.valueOf("nBep(*_ `_&F,Oh#*");
      assertNotNull(tag0);
      
      boolean boolean0 = tag0.preserveWhitespace();
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isData());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("k/DC<^W");
      boolean boolean0 = tag0.formatAsBlock();
      assertTrue(boolean0);
      assertFalse(tag0.isData());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tag tag0 = Tag.valueOf("G[(ak;J%");
      tag0.getName();
      assertTrue(tag0.formatAsBlock());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tag tag0 = Tag.valueOf("G[(ak;J%");
      tag0.toString();
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertTrue(tag0.formatAsBlock());
      assertTrue(tag0.isInline());
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Must be true");
      boolean boolean0 = tag0.isBlock();
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.formatAsBlock());
      assertFalse(boolean0);
      assertFalse(tag0.isData());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Tag tag0 = Tag.valueOf("wVjFpPi}r");
      boolean boolean0 = tag0.canContainBlock();
      assertTrue(tag0.formatAsBlock());
      assertTrue(tag0.isInline());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(boolean0);
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("link");
      boolean boolean0 = tag0.isInline();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tag tag0 = Tag.valueOf("l%1i6!,j9bcvy$!ru");
      boolean boolean0 = tag0.isInline();
      assertFalse(tag0.isData());
      assertTrue(boolean0);
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Tag tag0 = Tag.valueOf("h4");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tag tag0 = Tag.valueOf("h4");
      boolean boolean0 = tag0.isSelfClosing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tag tag0 = Tag.valueOf("k/DC<^W");
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isBlock());
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(boolean0);
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tag tag0 = Tag.valueOf("k/DC<^W");
      boolean boolean0 = tag0.isKnownTag();
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tag tag0 = Tag.valueOf("h4");
      boolean boolean0 = tag0.isKnownTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("-eta");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("meta");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Tag tag0 = Tag.valueOf("h4");
      Tag tag1 = Tag.valueOf("mol/_%+");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(tag1.isData());
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isSelfClosing());
      assertFalse(boolean0);
      assertTrue(tag1.isInline());
      assertFalse(tag1.preserveWhitespace());
      assertTrue(tag1.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Must be true");
      boolean boolean0 = tag0.equals(tag0);
      assertFalse(tag0.isData());
      assertTrue(tag0.canContainBlock());
      assertTrue(boolean0);
      assertFalse(tag0.isBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Tag tag0 = Tag.valueOf("h4");
      Object object0 = new Object();
      boolean boolean0 = tag0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Tag tag0 = Tag.valueOf("(<=T}");
      Tag tag1 = Tag.valueOf("mark");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.formatAsBlock());
      assertFalse(boolean0);
      assertFalse(tag0.isData());
      assertTrue(tag0.canContainBlock());
      assertTrue(tag0.isInline());
      assertFalse(tag0.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Tag tag0 = Tag.valueOf("link");
      Tag tag1 = Tag.valueOf("span");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Tag tag0 = Tag.valueOf("h4");
      Tag tag1 = Tag.valueOf("pre");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("n4&>'/fqo9f|w1]");
      Tag tag1 = Tag.valueOf("html");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
      assertTrue(tag0.canContainBlock());
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Must be true");
      Tag tag1 = Tag.valueOf("Must be true");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.isData());
      assertTrue(boolean0);
      assertTrue(tag1.formatAsBlock());
      assertTrue(tag1.canContainBlock());
      assertTrue(tag1.isInline());
      assertFalse(tag1.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Must be true");
      assertFalse(tag0.isSelfClosing());
      
      tag0.setSelfClosing();
      Tag tag1 = Tag.valueOf("Must be true");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Tag tag0 = Tag.valueOf("/D<W");
      Tag tag1 = Tag.valueOf("y\"");
      boolean boolean0 = tag1.equals(tag0);
      assertTrue(tag1.canContainBlock());
      assertFalse(tag1.isBlock());
      assertFalse(boolean0);
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.preserveWhitespace());
      assertFalse(tag1.isData());
      assertTrue(tag1.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Tag tag0 = Tag.valueOf("keygen");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Tag tag0 = Tag.valueOf("org.jsoup.parser.Tag");
      tag0.hashCode();
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Tag tag0 = Tag.valueOf("plaintext");
      tag0.hashCode();
  }
}
