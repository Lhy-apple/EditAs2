/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:58:00 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.parser.ParseSettings;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ParseSettings_ESTest extends ParseSettings_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.preserveCase;
      boolean boolean0 = parseSettings0.preserveTagCase();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ParseSettings parseSettings0 = new ParseSettings(false, true);
      parseSettings0.normalizeAttribute("");
      assertFalse(parseSettings0.preserveTagCase());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ParseSettings parseSettings0 = new ParseSettings(true, true);
      String string0 = parseSettings0.normalizeTag("tMy");
      assertEquals("tMy", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      String string0 = parseSettings0.normalizeTag("");
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      String string0 = parseSettings0.normalizeAttribute(",hT;r4P~");
      assertEquals(",ht;r4p~", string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.preserveCase;
      Attributes attributes0 = new Attributes();
      parseSettings0.normalizeAttributes(attributes0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Attributes attributes0 = new Attributes();
      Attributes attributes1 = parseSettings0.normalizeAttributes(attributes0);
      assertEquals(0, attributes1.size());
  }
}