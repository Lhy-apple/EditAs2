/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:52:02 GMT 2023
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
      Tag tag0 = Tag.valueOf("col");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
      assertTrue(tag0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tag tag0 = Tag.valueOf("*%\nJMhY4Qgw");
      assertNotNull(tag0);
      
      boolean boolean0 = tag0.preserveWhitespace();
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.formatAsBlock());
      assertFalse(boolean0);
      assertFalse(tag0.isBlock());
      assertFalse(tag0.isData());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("*%\nJMhY4Qw");
      boolean boolean0 = tag0.formatAsBlock();
      assertTrue(tag0.isInline());
      assertTrue(tag0.canContainBlock());
      assertTrue(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tag tag0 = Tag.valueOf(",kL0&.");
      tag0.getName();
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.isInline());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.canContainBlock());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tag tag0 = Tag.valueOf("t[)");
      tag0.toString();
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tag tag0 = Tag.valueOf("col");
      boolean boolean0 = tag0.isBlock();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Tag tag0 = Tag.valueOf(")p/i`ojp1;");
      boolean boolean0 = tag0.canContainBlock();
      assertFalse(tag0.isData());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isBlock());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("|%\nJM3hY4Qgw");
      assertFalse(tag0.isSelfClosing());
      
      tag0.setSelfClosing();
      Tag tag1 = Tag.valueOf("|%\nJM3hY4Qgw");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tag tag0 = Tag.valueOf("form");
      boolean boolean0 = tag0.isInline();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Tag tag0 = Tag.valueOf("&xq%jn%eq#\f yx(&>");
      boolean boolean0 = tag0.isInline();
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
      assertTrue(boolean0);
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isSelfClosing());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Tag tag0 = Tag.valueOf("kL0&D");
      boolean boolean0 = tag0.isData();
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tag tag0 = Tag.valueOf("col");
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tag tag0 = Tag.valueOf("*%JM3hY4Qw");
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(tag0.canContainBlock());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
      assertFalse(boolean0);
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tag tag0 = Tag.valueOf("h6");
      boolean boolean0 = tag0.isSelfClosing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tag tag0 = Tag.valueOf(".jcol");
      boolean boolean0 = tag0.isKnownTag();
      assertFalse(tag0.isData());
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isBlock());
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tag tag0 = Tag.valueOf("col");
      boolean boolean0 = tag0.isKnownTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("fv0e/w^6");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("col");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Tag tag0 = Tag.valueOf("|%\nJM3hY4Qgw");
      Tag tag1 = Tag.valueOf("|%\nJM3hY4Qgw");
      boolean boolean0 = tag0.equals(tag1);
      assertTrue(tag1.formatAsBlock());
      assertTrue(tag1.canContainBlock());
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.preserveWhitespace());
      assertTrue(tag1.isInline());
      assertFalse(tag1.isData());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Tag tag0 = Tag.valueOf("*%\nJM3hY4Qgw");
      boolean boolean0 = tag0.equals(tag0);
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
      assertTrue(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Tag tag0 = Tag.valueOf("plaintext");
      Object object0 = new Object();
      boolean boolean0 = tag0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Tag tag0 = Tag.valueOf("col");
      Tag tag1 = Tag.valueOf(".!k_8c/");
      boolean boolean0 = tag1.equals(tag0);
      assertTrue(tag1.canContainBlock());
      assertFalse(tag1.isData());
      assertFalse(tag1.preserveWhitespace());
      assertTrue(tag1.formatAsBlock());
      assertFalse(boolean0);
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.isBlock());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Tag tag0 = Tag.valueOf("col");
      Tag tag1 = Tag.valueOf("area");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Y*J8XyY");
      Tag tag1 = Tag.valueOf("title");
      boolean boolean0 = tag0.equals(tag1);
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.isData());
      assertFalse(boolean0);
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.isInline());
      assertFalse(tag0.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Tag tag0 = Tag.valueOf("_Vx");
      Tag tag1 = Tag.valueOf("caption");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(tag0.isData());
      assertTrue(tag0.formatAsBlock());
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertTrue(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Tag tag0 = Tag.valueOf("6^/(1|9{phir6i3ec");
      Tag tag1 = Tag.valueOf("o$md'5baa7>6dz");
      boolean boolean0 = tag0.equals(tag1);
      assertTrue(tag1.canContainBlock());
      assertFalse(boolean0);
      assertFalse(tag1.isData());
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.preserveWhitespace());
      assertFalse(tag1.isBlock());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Tag tag0 = Tag.valueOf("keygen");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Tag tag0 = Tag.valueOf("plaintext");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Tag tag0 = Tag.valueOf("kL0&D");
      tag0.hashCode();
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isSelfClosing());
      assertTrue(tag0.canContainBlock());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
  }
}
