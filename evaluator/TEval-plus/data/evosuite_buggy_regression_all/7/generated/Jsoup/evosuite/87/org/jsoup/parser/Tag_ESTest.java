/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:31:16 GMT 2023
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
      Tag tag0 = Tag.valueOf("}pv(Gw)h' pkK8~cC");
      assertNotNull(tag0);
      
      boolean boolean0 = tag0.preserveWhitespace();
      assertTrue(tag0.formatAsBlock());
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.canContainBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isData());
      assertFalse(tag0.isFormListed());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("pRNm");
      boolean boolean0 = tag0.formatAsBlock();
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(boolean0);
      assertTrue(tag0.isInline());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isFormSubmittable());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ParseSettings parseSettings0 = new ParseSettings(true, true);
      Tag tag0 = Tag.valueOf("1yak5\"A0bDPXz]|8QE&", parseSettings0);
      tag0.getName();
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isData());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("lA$QD)vip", parseSettings0);
      tag0.toString();
      assertFalse(tag0.isData());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isBlock());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tag tag0 = Tag.valueOf("t.j)_(MTC");
      boolean boolean0 = tag0.isFormListed();
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isData());
      assertFalse(tag0.canContainBlock());
      assertFalse(boolean0);
      assertTrue(tag0.formatAsBlock());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Tag tag0 = Tag.valueOf("param");
      boolean boolean0 = tag0.isBlock();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tag tag0 = Tag.valueOf("param");
      boolean boolean0 = tag0.canContainBlock();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tag tag0 = Tag.valueOf("}pv(Gw)h' pkK8~cC");
      boolean boolean0 = tag0.isFormSubmittable();
      assertFalse(tag0.isData());
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormListed());
      assertTrue(tag0.isInline());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("P", parseSettings0);
      assertFalse(tag0.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Tag tag0 = Tag.valueOf("hgroup");
      boolean boolean0 = tag0.isInline();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tag tag0 = Tag.valueOf("a");
      boolean boolean0 = tag0.isInline();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Rg/lYuN");
      boolean boolean0 = tag0.isData();
      assertFalse(boolean0);
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.canContainBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isFormListed());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tag tag0 = Tag.valueOf("param");
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tag tag0 = Tag.valueOf("LmXl");
      boolean boolean0 = tag0.isSelfClosing();
      assertFalse(tag0.preserveWhitespace());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isData());
      assertFalse(tag0.isFormListed());
      assertFalse(boolean0);
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tag tag0 = Tag.valueOf("LmXl");
      assertFalse(tag0.isSelfClosing());
      
      tag0.setSelfClosing();
      boolean boolean0 = tag0.isSelfClosing();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Tag tag0 = Tag.valueOf("rm");
      boolean boolean0 = tag0.isKnownTag();
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.canContainBlock());
      assertFalse(tag0.isData());
      assertFalse(tag0.isFormListed());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Tag tag0 = Tag.valueOf("bgsound");
      boolean boolean0 = tag0.isKnownTag();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("paHam");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      boolean boolean0 = Tag.isKnownTag("title");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Rg/lYuN");
      Tag tag1 = Tag.valueOf("Rg/lYuN");
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(tag1.isData());
      assertTrue(tag1.formatAsBlock());
      assertFalse(tag1.preserveWhitespace());
      assertTrue(boolean0);
      assertFalse(tag1.isFormSubmittable());
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.isFormListed());
      assertFalse(tag1.isBlock());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Tag tag0 = Tag.valueOf("param");
      boolean boolean0 = tag0.equals(tag0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Rg/lYuN");
      boolean boolean0 = tag0.equals("plaintext");
      assertFalse(boolean0);
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormListed());
      assertFalse(tag0.isData());
      assertFalse(tag0.preserveWhitespace());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.canContainBlock());
      assertFalse(tag0.isFormSubmittable());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Rg/lYuN");
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag1 = Tag.valueOf("Rg/lYuN", parseSettings0);
      boolean boolean0 = tag1.equals(tag0);
      assertFalse(tag1.isSelfClosing());
      assertFalse(tag1.isFormListed());
      assertTrue(tag1.formatAsBlock());
      assertFalse(boolean0);
      assertFalse(tag1.isFormSubmittable());
      assertFalse(tag1.isData());
      assertFalse(tag1.isBlock());
      assertFalse(tag1.preserveWhitespace());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Tag tag0 = Tag.valueOf("xml");
      Tag tag1 = Tag.valueOf("xml");
      assertFalse(tag1.isSelfClosing());
      
      tag1.setSelfClosing();
      boolean boolean0 = tag0.equals(tag1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Rg/lYuN");
      tag0.hashCode();
      assertFalse(tag0.isSelfClosing());
      assertFalse(tag0.isFormListed());
      assertTrue(tag0.formatAsBlock());
      assertFalse(tag0.isFormSubmittable());
      assertFalse(tag0.canContainBlock());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("plaintext", parseSettings0);
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Tag tag0 = Tag.valueOf("object");
      tag0.hashCode();
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Tag tag0 = Tag.valueOf("base");
      tag0.hashCode();
  }
}