/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:20:13 GMT 2023
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
      Tag tag0 = Tag.valueOf("meta");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tag tag0 = Tag.valueOf("jW");
      assertNotNull(tag0);
      
      boolean boolean0 = tag0.preserveWhitespace();
      assertFalse(boolean0);
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isData());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.canContainBlock());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isSelfClosing());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("basefont");
      boolean boolean0 = tag0.formatAsBlock();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tag tag0 = Tag.valueOf("`fI<-XO'wzjT\"|L^'<");
      tag0.getName();
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.canContainBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertFalse(tag0.isFormListed());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tag tag0 = Tag.valueOf("0WjxN./Q");
      tag0.toString();
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      boolean boolean0 = tag0.isFormListed();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Tag tag0 = Tag.valueOf("7bcHvGq!wV8&Xq");
      boolean boolean0 = tag0.isBlock();
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isFormSubmittable());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("1S%sB%lTb1&vY");
      boolean boolean0 = tag0.canContainBlock();
      assertFalse(boolean0);
      assertFalse(tag0.isData());
      assertFalse(tag0.isFormListed());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      boolean boolean0 = tag0.isFormSubmittable();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Tag tag0 = Tag.valueOf("k!e");
      assertFalse(tag0.isSelfClosing());
      
      tag0.setSelfClosing();
      Tag tag1 = Tag.valueOf("k!e");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("S", parseSettings0);
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      boolean boolean0 = tag0.isInline();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tag tag0 = Tag.valueOf("IAt>");
      boolean boolean0 = tag0.isInline();
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(boolean0);
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isData());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Vz.{Nw7");
      boolean boolean0 = tag0.isSelfClosing();
      assertFalse(boolean0);
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Vz.{Nw7");
      assertFalse(tag0.isSelfClosing());
      
      tag0.setSelfClosing();
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Tag tag0 = Tag.valueOf("1rx>!p&]<j=");
      boolean boolean0 = tag0.isKnownTag();
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(boolean0);
      assertTrue(tag0.isInline());
      assertFalse(tag0.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      boolean boolean0 = tag0.isKnownTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("k!e");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("meta");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Tag tag0 = Tag.valueOf("%~{2,O}jdQ<XDB");
      Tag tag1 = Tag.valueOf("jH-");
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(tag1.isData());
      assertFalse(tag1.preserveWhitespace());
      assertFalse(boolean0);
      assertFalse(tag1.isFormListed());
      assertFalse(tag1.isSelfClosing());
      assertTrue(tag1.isInline());
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isFormSubmittable());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Tag tag0 = Tag.valueOf("k!e");
      boolean boolean0 = tag0.equals(tag0);
      assertTrue(boolean0);
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isData());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Tag tag0 = Tag.valueOf("n-#,1`");
      boolean boolean0 = tag0.equals("n-#,1`");
      assertFalse(tag0.preserveWhitespace());
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isData());
      assertFalse(tag0.canContainBlock());
      assertFalse(tag0.isFormSubmittable());
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("k!e");
      Tag tag1 = Tag.valueOf("k!e");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(tag1.preserveWhitespace());
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.isFormSubmittable());
      assertTrue(boolean0);
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.isBlock());
      assertFalse(tag1.isData());
      assertFalse(tag1.isFormListed());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.preserveCase;
      Tag tag0 = Tag.valueOf("object", parseSettings0);
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Tag tag0 = Tag.valueOf("meta");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Tag tag0 = Tag.valueOf("k!e");
      tag0.hashCode();
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.isInline());
      assertFalse(tag0.isSelfClosing());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("pre", parseSettings0);
      tag0.hashCode();
  }
}