/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:43:41 GMT 2023
 */

package org.apache.commons.lang;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.apache.commons.lang.LocaleUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LocaleUtils_ESTest extends LocaleUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LocaleUtils localeUtils0 = new LocaleUtils();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      List list0 = LocaleUtils.localeLookupList((Locale) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("zn_OG_p`T");
      assertNotNull(locale0);
      assertEquals("zn_OG_p`T", locale0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale((String) null);
      assertNull(locale0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("zn_OT");
      assertEquals("zn_OT", locale0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("W a0%");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: W a0%
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("~pQbje|_\"kU1&pFu18");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ~pQbje|_\"kU1&pFu18
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("jLUv9bQ");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: jLUv9bQ
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("s|`*!l+PJW");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: s|`*!l+PJW
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("ss`*!l+PJW");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ss`*!l+PJW
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("zn_\u0011Gu8&");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: zn_\u0011Gu8&
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("zn_O$$2u&");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: zn_O$$2u&
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("zn_Ow$2u&");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: zn_Ow$2u&
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("zn_OGa&");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: zn_OGa&
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Locale locale0 = new Locale("Jf", "Jf", "h_A?#1!d{(Ik");
      List list0 = LocaleUtils.localeLookupList(locale0, locale0);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Locale locale0 = Locale.ENGLISH;
      List list0 = LocaleUtils.localeLookupList(locale0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Locale locale0 = Locale.CANADA_FRENCH;
      List list0 = LocaleUtils.localeLookupList(locale0, (Locale) null);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      LocaleUtils.availableLocaleSet();
      Set set0 = LocaleUtils.availableLocaleSet();
      assertEquals(160, set0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Locale locale0 = new Locale("Jf", "Jf", "h_A?#1!d{(Ik");
      boolean boolean0 = LocaleUtils.isAvailableLocale(locale0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Locale locale0 = Locale.US;
      boolean boolean0 = LocaleUtils.isAvailableLocale(locale0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Set set0 = LocaleUtils.availableLocaleSet();
      assertNotNull(set0);
      
      List list0 = LocaleUtils.languagesByCountry("");
      assertEquals(46, list0.size());
      assertNotNull(list0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      LocaleUtils.languagesByCountry("6&V}e|p(g9)Z9rT3n");
      LocaleUtils.countriesByLanguage("TH");
      LocaleUtils.countriesByLanguage("DO");
      LocaleUtils.languagesByCountry((String) null);
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("TH");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: TH
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      LocaleUtils.languagesByCountry("TH");
      String string0 = "CH";
      LocaleUtils.countriesByLanguage("CH");
      LocaleUtils.languagesByCountry("");
      String string1 = "";
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: 
         //
         verifyException("org.apache.commons.lang.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      List list0 = LocaleUtils.countriesByLanguage((String) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      LocaleUtils.countriesByLanguage("pl");
      Locale locale0 = LocaleUtils.toLocale("pl");
      assertEquals("pl", locale0.toString());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      LocaleUtils.countriesByLanguage("pl");
      List list0 = LocaleUtils.countriesByLanguage("ja");
      assertEquals(1, list0.size());
  }
}