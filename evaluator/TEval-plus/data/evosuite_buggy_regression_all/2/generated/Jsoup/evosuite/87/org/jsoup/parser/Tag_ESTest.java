/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:16:46 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.parser.ParseSettings;
import org.jsoup.parser.Tag;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Tag_ESTest extends Tag_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Tag tag0 = Tag.valueOf("param");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tag tag0 = Tag.valueOf("html");
      boolean boolean0 = tag0.preserveWhitespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("<2QveQB3?x'W$r");
      assertNotNull(tag0);
      
      boolean boolean0 = tag0.formatAsBlock();
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isSelfClosing());
      assertTrue(boolean0);
      assertFalse(tag0.canContainBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tag tag0 = Tag.valueOf("header");
      String string0 = tag0.getName();
      assertEquals("header", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("formnovalidate", parseSettings0);
      tag0.toString();
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.canContainBlock());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormSubmittable());
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tag tag0 = Tag.valueOf("1ZQ{?");
      boolean boolean0 = tag0.isFormListed();
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isBlock());
      assertTrue(tag0.formatAsBlock());
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Tag tag0 = Tag.valueOf("keygen");
      boolean boolean0 = tag0.isBlock();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("-M(c)FcAb9Ou<");
      boolean boolean0 = tag0.canContainBlock();
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormListed());
      assertFalse(boolean0);
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormSubmittable());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tag tag0 = Tag.valueOf("<2QveQB3?x'W$r");
      boolean boolean0 = tag0.isFormSubmittable();
      assertFalse(tag0.isData());
      assertFalse(boolean0);
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("PLAINTEXT", parseSettings0);
      assertFalse(tag0.isSelfClosing());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Tag tag0 = Tag.valueOf("header");
      boolean boolean0 = tag0.isInline();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tag tag0 = Tag.valueOf("!towC U|,FI-a3v9.");
      boolean boolean0 = tag0.isInline();
      assertTrue(tag0.formatAsBlock());
      assertTrue(boolean0);
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isData());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tag tag0 = Tag.valueOf("<2QveQB3?x'W$r");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tag tag0 = Tag.valueOf("param");
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tag tag0 = Tag.valueOf("!&36qjF");
      boolean boolean0 = tag0.isSelfClosing();
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isFormListed());
      assertFalse(boolean0);
      assertFalse(tag0.isData());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tag tag0 = Tag.valueOf("html");
      boolean boolean0 = tag0.isSelfClosing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Tag tag0 = Tag.valueOf(":aram");
      boolean boolean0 = tag0.isKnownTag();
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormSubmittable());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertFalse(boolean0);
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Tag tag0 = Tag.valueOf("keygen");
      boolean boolean0 = tag0.isKnownTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("1ZQ{?");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("h6");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Tag tag0 = Tag.valueOf("<2QveQB3?x'W$r");
      Tag tag1 = Tag.valueOf("org.jsoup.nodes.Attributes$1");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.isFormListed());
      assertFalse(boolean0);
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isFormSubmittable());
      assertFalse(tag1.isData());
      assertFalse(tag1.preserveWhitespace());
      assertTrue(tag1.isInline());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Tag tag0 = Tag.valueOf("param");
      boolean boolean0 = tag0.equals(tag0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Tag tag0 = Tag.valueOf("title");
      Object object0 = new Object();
      boolean boolean0 = tag0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("=oa+3R,RZ/w2PkH", parseSettings0);
      Tag tag1 = Tag.valueOf("=oa+3R,RZ/w2PkH", parseSettings0);
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.isFormListed());
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isFormSubmittable());
      assertFalse(tag1.isData());
      assertFalse(tag1.preserveWhitespace());
      assertTrue(boolean0);
      assertFalse(tag1.isBlock());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Tag tag0 = Tag.valueOf("e8V;`#TUZ7k;MEyhG6");
      assertFalse(tag0.isSelfClosing());
      
      tag0.setSelfClosing();
      Tag tag1 = Tag.valueOf("e8V;`#TUZ7k;MEyhG6");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Tag tag0 = Tag.valueOf("title");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Tag tag0 = Tag.valueOf("nav");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Tag tag0 = Tag.valueOf("bdo");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Tag tag0 = Tag.valueOf("keygen");
      tag0.hashCode();
  }
}